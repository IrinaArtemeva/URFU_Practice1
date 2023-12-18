
import streamlit as st
import easyocr

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

    st.write('Раздел в разработке!')
    st.image('http://www.opsa.info/img/innovations/v-nastoyaschee-vremya-razdel-nahoditsya-v-razrabotke.jpg')