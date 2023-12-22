import json

import newspaper
import bs4
import requests

class Article:

    def __init__(self, title='', link='', authors=''):
        self.__title = title
        self.__link = link
        self.__authors = authors
        self.__year = 0
        self.__rsci = False
        self.__vak = False
        self.__scopus = False

    def check_filter(self, filter):
        if filter == 22 and not self.__rsci or filter == 8 and not self.__vak or filter == 2 and not self.__scopus:
            return False
        return True

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, value):
        self.__title = value

    @property
    def link(self):
        return self.__link

    @link.setter
    def link(self, value):
        self.__link = value

    @property
    def authors(self):
        return self.__authors

    @authors.setter
    def authors(self, value):
        self.__authors = value

    @property
    def year(self):
        return self.__year

    @year.setter
    def year(self, value):
        self.__year = value

    @property
    def rsci(self):
        return self.__rsci

    @rsci.setter
    def rsci(self, value):
        self.__rsci = value

    @property
    def vak(self):
        return self.__vak

    @vak.setter
    def vak(self, value):
        self.__vak = value

    @property
    def scopus(self):
        return self.__scopus

    @scopus.setter
    def scopus(self, value):
        self.__scopus = value

TERM_LINK = '"link":'
TERM_FOUND = '"found":'

API = 'https://cyberleninka.ru/api/search'
URL = 'https://cyberleninka.ru'

REQUEST_BODY = {
    'mode': 'articles',
    'size': 10,
}

ARTICLES_PER_PAGE = 10


class Searcher:

    def __try_parse_article(self, link, article):
        response_page = requests.get(link)
        bs_page = bs4.BeautifulSoup(response_page.text, 'html.parser')
        try:
            article.title = bs_page.i.text
        except:
            print('done')

        authors = bs_page.find('h2', {'class': 'right-title'}).span.text
        article.authors = authors[authors.find('—') + 2:]

        labels = bs_page.find('div', {'class': 'labels'})
        article.year = int(labels.time.text)

        rsci = bs_page.find('div', {'class': 'label rsci'})
        if rsci:
            article.rsci = True

        vak = bs_page.find('div', {'class': 'label vak'})
        if vak:
            article.vak = True

        scopus = bs_page.find('div', {'class': 'label scopus'})
        if scopus:
            article.scopus = True

    def parse_article_page(self, link):
        article = Article(link=link)

        try:
            self.__try_parse_article(link, article)

        except requests.HTTPError or AttributeError as e:
            print(e)

        return article

    def __try_parse_request(self, body, filters, results, articles_per_page=ARTICLES_PER_PAGE):
        response_json = requests.post(API, data=json.dumps(body)).text

        for article_num in range(articles_per_page):
            response_json = response_json[response_json.find(TERM_LINK) + len(TERM_LINK) + 1:]
            article_link = response_json[:response_json.find('"')]
            article = self.parse_article_page(URL + article_link)
            if filters is None or article.check_filter(filters[0]):
                results.append(article)
            response_json = response_json[response_json.find('}'):]

    def search_articles(self, keywords, max_page, filters=None):
        results = []
        found = 0
        body = {
            'mode': 'articles',
            'size': 10,
            'q': keywords}

        if filters:
            body = body | {'catalogs': filters}

        body['from'] = 0

        try:
            response_json = requests.post(API, data=json.dumps(body)).text
            response_found = response_json[response_json.find(TERM_FOUND) + len(TERM_FOUND):]
            #print(response_found)
            #print(response_json)
            resp_data = json.loads(response_json)
            relevant = resp_data["articles"]
            for i in relevant:

                def special_char_fix(string):
                    string = list(string)
                    for pl, char in enumerate(string):
                        if char == '\\':
                            val = ''.join([string[pl + k + 2] for k in range(4)])
                            for k in range(5):
                                string.pop(pl)
                            string[pl] = str(chr(int(val, 16)))
                    return ''.join(string)

                print("Название статьи:" + ' ' + (special_char_fix(i["name"])) + ' ' + 'Аннотация:' + ' ' + i["annotation"] + ' ' + i["link"], end='\n')
            found = int(response_found[:response_found.find(',')])

        except requests.HTTPError or AttributeError as e:
            print(e)

        for page in range(max_page):
            try:
                if found >= ARTICLES_PER_PAGE:
                    body['from'] += 10 * page
                    self.__try_parse_request(body, filters, results)
                    found -= ARTICLES_PER_PAGE
                else:
                    body['from'] += 10 * (page - 1)
                    self.__try_parse_request(body, filters, results, found)
                    return results

            except requests.HTTPError as e:
                print(e)
                return []

        return results




if __name__ == '__main__':
    keywords = input('Input your keywords.\n').lower()
    pr = Searcher()
    pr.search_articles(keywords, 10)
    #print(res)
    #for r in res:
    #    print(r)
