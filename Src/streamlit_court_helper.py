
import streamlit as st
import easyocr
import json
import bs4
import requests


# Установка конфигурации страницы
st.set_page_config(
page_title="Здесь будет красивое название",
page_icon="🧊")

# Изменение дизайна страницы
page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://i.artfile.ru/1920x1080_927461_[www.ArtFile.ru].jpg");
  background-size: cover;
}
[data-testid="baseButton-secondary"]{
background-color: rgb(176 96 255 / 70%);
}
[role="tablist"]{
background-color: rgb(210 164 255 / 90%);
border-radius: 10px;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
.st-b1{
  margin: auto;
}
[data-testid="block-container"]
 { background-color: rgba(255, 255, 255, 0.4)
}
.st-cc {
    margin: auto;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)

# Заголовок страницы
st.header('Приветствуем! Здесь вы можете отсканировать текст с изображения или найти информацию по ключевым словам.')

# Создание отдельных вкладок для функционала
tabs = st.tabs(["Распознавание", "Поиск"])

# Содержание вкладки распознавания текста
with tabs[0]:
    @st.cache_resource()
    def easyocr_recognition(img):
        return easyocr.Reader(["ru", "en", "uk"]).readtext(img, detail=0, paragraph=True, text_threshold=0.8)

    # распознавание с помощью easyocr, параметры: отключена детализация вывода,
    # включены параграфы и установлена точность текста

    # сохранение текста в текстовый файл

    def load_image():
        """Создание формы для загрузки изображения"""
        # Форма для загрузки изображения средствами Streamlit
        uploaded_file = st.file_uploader(
            label='Выберите изображение, содержащее текст, для распознавания.\n Формат загружаемого файла .jpg .png')
        if uploaded_file is not None:
            # Получение загруженного изображения
            img = uploaded_file.getvalue()
            # Показ загруженного изображения на Web-странице средствами Streamlit
            st.image(img)
            return img
        else:
            return st.write('Вы ничего не загрузили!')

    # Выводим заголовок страницы средствами Streamlit
    st.header('Это простой инструмент для распознавания изображений, содержащих текст')

    # Вызываем функцию создания формы загрузки изображения

    img = load_image()

    result = st.button('Распознать документ из файла')
    if result:
        try:
            text_result = easyocr_recognition(img)
            st.write(text_result)
        except:
            st.write('Отсутствует файл для распознавания или неверный формат файла!')

# Содержание вкладки поиска по ключевым словам
with tabs[1]:
    st.header('Это простой инструмент для поиска информации по ключевым словам')
# тело функции поиска по ключевым словам
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
    '   size': 10,
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

                    st.write("Название статьи:" + ' ' + (special_char_fix(i["name"])) + ' ' + 'Аннотация:' + ' ' + i["annotation"] + ' ' + i["link"], end='\n')
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




#if __name__ == '__main__':
    keywords = st.text_input(('Input your keywords.\n').lower())
    pr = Searcher()
    pr.search_articles(keywords, 10)

    #st.write('Раздел в разработке!')
    #st.image('http://www.opsa.info/img/innovations/v-nastoyaschee-vremya-razdel-nahoditsya-v-razrabotke.jpg')
