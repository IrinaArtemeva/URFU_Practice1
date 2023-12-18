
import streamlit as st
import easyocr




@st.cache_resource()
def easyocr_recognition(img):
    return easyocr.Reader(["ru", "en"]).readtext(img, detail=0, paragraph=True, text_threshold=0.8)


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

st.title('Это простой инструмент для распознавания изображений, содержащих текст')

# Вызываем функцию создания формы загрузки изображения

img = load_image()

result = st.button('Распознать документ из файла')
if result:
    try:
        text_result = easyocr_recognition(img)
        st.write(text_result)
    except:
        st.write('Отсутствует файл для распознавания или неверный формат файла!')