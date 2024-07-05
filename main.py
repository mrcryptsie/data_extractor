import streamlit as st
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

def plt_imshow(title, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption=title, use_column_width=True)

def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 11))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    tableCnt = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(tableCnt)
    table = image[y:y + h, x:x + w]
    options = "--psm 6"
    results = pytesseract.image_to_data(cv2.cvtColor(table, cv2.COLOR_BGR2RGB), config=options, output_type=Output.DICT)
    coords = []
    ocrText = []
    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]
        conf = int(float(results["conf"][i]))
        if conf > 1:
            coords.append((x, y, w, h))
            ocrText.append(text)
    xCoords = [(c[0], 0) for c in coords]
    clustering = AgglomerativeClustering(n_clusters=None, metric="manhattan", linkage="complete", distance_threshold=25.0)
    clustering.fit(xCoords)
    sortedClusters = []
    for l in np.unique(clustering.labels_):
        idxs = np.where(clustering.labels_ == l)[0]
        if len(idxs) > 2:
            avg = np.average([coords[i][0] for i in idxs])
            sortedClusters.append((l, avg))
    sortedClusters.sort(key=lambda x: x[1])
    df = pd.DataFrame()
    for (l, _) in sortedClusters:
        idxs = np.where(clustering.labels_ == l)[0]
        yCoords = [coords[i][1] for i in idxs]
        sortedIdxs = idxs[np.argsort(yCoords)]
        color = np.random.randint(0, 255, size=(3,), dtype="int")
        color = [int(c) for c in color]
        for i in sortedIdxs:
            (x, y, w, h) = coords[i]
            cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)
        cols = [ocrText[i].strip() for i in sortedIdxs]
        currentDF = pd.DataFrame({cols[0]: cols[1:]})
        df = pd.concat([df, currentDF], axis=1)
    df.fillna("", inplace=True)
    return df

st.title("Extraction de texte d'une image")
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    plt_imshow("Image téléchargée", image)
    df = extract_text_from_image(image)
    st.write("Résultat de l'extraction de texte :")
    st.dataframe(df)