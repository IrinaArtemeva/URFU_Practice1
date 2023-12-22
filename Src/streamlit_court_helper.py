
import streamlit as st
import easyocr
import json
import bs4
import requests
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np
import PyPDF2


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
tabs = st.tabs(["Распознавание", "Поиск", "Экстрактор"])

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





    keywords = st.text_input(('Введите ключевые слова.\n').lower())
    result1 = st.button('Вывести список релевантных статей')
    if result1:
        try:
            pr = Searcher()
            pr.search_articles(keywords, 10)
        except:
            st.write('вы не начали поиск!')

with tabs[2]:
    st.header('Это простой инструмент для извлечения ключевых слов из текстового документа')
    # Для анализа структуры PDF и извлечения текста
    from pdfminer.high_level import extract_pages, extract_text
    from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
    # Для извлечения текста из таблиц в PDF
    import pdfplumber


    # Создаем класс, определяющий пайплайн выделения ключевых слов из текста с использованием предобученной модели
    class KeyphraseExtractionPipeline(TokenClassificationPipeline):
        def __init__(self, model, *args, **kwargs):
            super().__init__(
                model=AutoModelForTokenClassification.from_pretrained(model),
                tokenizer=AutoTokenizer.from_pretrained(model),
                *args,
                **kwargs
            )

        def postprocess(self, all_outputs):
            results = super().postprocess(
                all_outputs=all_outputs,
                aggregation_strategy=AggregationStrategy.SIMPLE,
            )
            return np.unique([result.get("word").strip() for result in results])

    def filereader(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            draft = file.read()
        return draft


    def text_extraction(element):
        # Извлекаем текст из вложенного текстового элемента
        line_text = element.get_text()

        # Находим форматы текста
        # Инициализируем список со всеми форматами, встречающимися в строке текста
        line_formats = []
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                # Итеративно обходим каждый символ в строке текста
                for character in text_line:
                    if isinstance(character, LTChar):
                        # Добавляем к символу название шрифта
                        line_formats.append(character.fontname)
                        # Добавляем к символу размер шрифта
                        line_formats.append(character.size)
        # Находим уникальные размеры и названия шрифтов в строке
        format_per_line = list(set(line_formats))

        # Возвращаем кортеж с текстом в каждой строке вместе с его форматом
        return (line_text, format_per_line)

    def extract_table(pdf_path, page_num, table_num):
        # Открываем файл pdf
        pdf = pdfplumber.open(pdf_path)
        # Находим исследуемую страницу
        table_page = pdf.pages[page_num]
        # Извлекаем соответствующую таблицу
        table = table_page.extract_tables()[table_num]
        return table

    # Преобразуем таблицу в соответствующий формат
    def table_converter(table):
        table_string = ''
        # Итеративно обходим каждую строку в таблице
        for row_num in range(len(table)):
            row = table[row_num]
            # Удаляем разрыв строки из текста с переносом
            cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
            # Преобразуем таблицу в строку
            table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')
        # Удаляем последний разрыв строки
        table_string = table_string[:-1]
        return table_string

    def crop_image(element, pageObj):
        # Получаем координаты для вырезания изображения из PDF
        [image_left, image_top, image_right, image_bottom] = [element.x0,element.y0,element.x1,element.y1]
        # Обрезаем страницу по координатам (left, bottom, right, top)
        pageObj.mediabox.lower_left = (image_left, image_bottom)
        pageObj.mediabox.upper_right = (image_right, image_top)
        # Сохраняем обрезанную страницу в новый PDF
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_writer.add_page(pageObj)
        # Сохраняем обрезанный PDF в новый файл
        with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
            cropped_pdf_writer.write(cropped_pdf_file)

    # Создаём функцию для считывания текста из изображений
    def image_to_text(image_path):
        # Считываем изображение
        img = Image.open(image_path)
        # Извлекаем текст из изображения
        text = pytesseract.image_to_string(img)
        return text



    model_name = "ml6team/keyphrase-extraction-kbir-inspec"
    extractor = KeyphraseExtractionPipeline(model=model_name)


    file_path = st.file_uploader(
                label='Выберите изображение, содержащее текст, для распознавания.\n Формат загружаемого файла .txt, .pdf')

    indent = '.pdf'

    pdf_path = file_path

    # создаём объект файла PDF
    pdfFileObj = open(pdf_path, 'rb')
    # создаём объект считывателя PDF
    pdfReaded = PyPDF2.PdfReader(pdfFileObj)

    # Создаём словарь для извлечения текста из каждого изображения
    text_per_page = {}
    # Извлекаем страницы из PDF
    for pagenum, page in enumerate(extract_pages(pdf_path)):

        # Инициализируем переменные, необходимые для извлечения текста со страницы
        pageObj = pdfReaded.pages[pagenum]
        page_text = []
        line_format = []
        text_from_images = []
        text_from_tables = []
        page_content = []
        # Инициализируем количество исследованных таблиц
        table_num = 0
        first_element = True
        table_extraction_flag = False
        # Открываем файл pdf
        pdf = pdfplumber.open(pdf_path)
        # Находим исследуемую страницу
        page_tables = pdf.pages[pagenum]
        # Находим количество таблиц на странице
        tables = page_tables.find_tables()

        # Находим все элементы
        page_elements = [(element.y1, element) for element in page._objs]
        # Сортируем все элементы по порядку нахождения на странице
        page_elements.sort(key=lambda a: a[0], reverse=True)

        # Находим элементы, составляющие страницу
        for i, component in enumerate(page_elements):
            # Извлекаем положение верхнего края элемента в PDF
            pos = component[0]
            # Извлекаем элемент структуры страницы
            element = component[1]

            # Проверяем, является ли элемент текстовым
            if isinstance(element, LTTextContainer):
                # Проверяем, находится ли текст в таблице
                if table_extraction_flag == False:
                    # Используем функцию извлечения текста и формата для каждого текстового элемента
                    (line_text, format_per_line) = text_extraction(element)
                    # Добавляем текст каждой строки к тексту страницы
                    page_text.append(line_text)
                    # Добавляем формат каждой строки, содержащей текст
                    line_format.append(format_per_line)
                    page_content.append(line_text)
                else:
                    # Пропускаем текст, находящийся в таблице
                    pass

            # Проверяем элементы на наличие таблиц
            if isinstance(element, LTRect):
                # Если первый прямоугольный элемент
                if first_element == True and (table_num + 1) <= len(tables):
                    # Находим ограничивающий прямоугольник таблицы
                    lower_side = page.bbox[3] - tables[table_num].bbox[3]
                    upper_side = element.y1
                    # Извлекаем информацию из таблицы
                    table = extract_table(pdf_path, pagenum, table_num)
                    # Преобразуем информацию таблицы в формат структурированной строки
                    table_string = table_converter(table)
                    # Добавляем строку таблицы в список
                    text_from_tables.append(table_string)
                    page_content.append(table_string)
                    # Устанавливаем флаг True, чтобы избежать повторения содержимого
                    table_extraction_flag = True
                    # Преобразуем в другой элемент
                    first_element = False
                    # Добавляем условное обозначение в списки текста и формата
                    page_text.append('table')
                    line_format.append('table')

                # Проверяем, извлекли ли мы уже таблицы из этой страницы
                if element.y0 >= lower_side and element.y1 <= upper_side:
                    pass
                elif not isinstance(page_elements[i + 1][1], LTRect):
                    table_extraction_flag = False
                    first_element = True
                    table_num += 1

        # Создаём ключ для словаря
        dctkey = 'Page_' + str(pagenum)
        # Добавляем список списков как значение ключа страницы
        text_per_page[dctkey] = [page_text, line_format, text_from_images, text_from_tables, page_content]

    # Закрываем объект файла pdf
    pdfFileObj.close()

    # Удаляем содержимое страницы
    result = ''.join(text_per_page['Page_0'][4])
    print(result)
    text = result.replace("\n", " ")
    keyphrases = extractor(text)
    print(keyphrases)
    else:
        text = filereader(file_path).replace("\n", " ")
        keyphrases = extractor(text)
        print(keyphrases)