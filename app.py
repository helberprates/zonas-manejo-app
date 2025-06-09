import streamlit as st
import ee
import tempfile
import geopandas as gpd
from zipfile import ZipFile
import os
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import numpy as np
import datetime
import geemap.foliumap as geemap
from fpdf import FPDF

# Autenticar no Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

st.set_page_config(page_title="Zonas de Manejo", layout="wide")
st.title("🛰️ Gerador de Zonas de Manejo via NDVI, NDRE, SBI e Altimetria")

# --- Upload do Shapefile ---
with st.expander("📂 Upload do shapefile (.zip com .shp, .dbf, .shx, .prj)"):
    uploaded_zip = st.file_uploader("Envie o arquivo .zip contendo o shapefile", type="zip")
    area_geom = None

    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "area.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if shp_files:
                gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                gdf = gdf.to_crs(epsg=4326)
                area_geom = ee.Geometry.Polygon(gdf.geometry.iloc[0].exterior.coords[:])
                st.success("Área carregada com sucesso!")
            else:
                st.error("Nenhum .shp encontrado no .zip enviado.")

# --- Parâmetros ---
st.sidebar.header("⚙️ Parâmetros da Análise")
data_inicio = st.sidebar.date_input("Data inicial", datetime.date(2025, 1, 1))
data_fim = st.sidebar.date_input("Data final", datetime.date(2025, 3, 31))
k_zonas = st.sidebar.slider("Número de zonas de manejo", min_value=2, max_value=7, value=4)

# --- Processar quando clicar ---
if st.button("🚀 Gerar Zonas de Manejo") and area_geom:
    with st.spinner("Buscando imagens no GEE e processando..."):

        def preparar_imagem(img):
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndre = img.normalizedDifference(['B8', 'B5']).rename('NDRE')
            sbi = img.expression('sqrt(B2**2 + B3**2)', {
                'B2': img.select('B2'),
                'B3': img.select('B3')
            }).rename('SBI')
            return img.addBands([ndvi, ndre, sbi])

        colecao = ee.ImageCollection('COPERNICUS/S2_SR')             .filterBounds(area_geom)             .filterDate(str(data_inicio), str(data_fim))             .map(preparar_imagem)

        imagem_median = colecao.median()

        srtm = ee.Image("USGS/SRTMGL1_003")
        imagem_final = imagem_median.addBands(srtm.rename('ALT'))

        amostras = imagem_final.select(['NDVI', 'NDRE', 'SBI', 'ALT']).sample(
            region=area_geom,
            scale=10,
            numPixels=5000,
            geometries=True
        ).getInfo()

        dados = []
        coords = []
        for feat in amostras['features']:
            props = feat['properties']
            dados.append([props['NDVI'], props['NDRE'], props['SBI'], props['ALT']])
            coords.append(feat['geometry']['coordinates'])

        kmeans = KMeans(n_clusters=k_zonas, random_state=42).fit(dados)
        labels = kmeans.labels_

        valores = [ee.Feature(ee.Geometry.Point(c), {"zona": int(z)}) for c, z in zip(coords, labels)]
        feature_collection = ee.FeatureCollection(valores)

        zonas_raster = feature_collection.reduceToImage(properties=['zona'], reducer=ee.Reducer.first())

        task = ee.batch.Export.image.toDrive(
            image=zonas_raster.toByte(),
            description='zonas_manejo_export',
            folder='gee_exports',
            region=area_geom.bounds().getInfo()['coordinates'],
            scale=10,
            maxPixels=1e13
        )
        task.start()

        m = folium.Map(location=coords[0][::-1], zoom_start=15)
        for i, coord in enumerate(coords):
            folium.CircleMarker(
                location=coord[::-1],
                radius=2,
                fill=True,
                color=f"#{(hash(i)%0xFFFFFF):06x}",
                tooltip=f"Zona {labels[i]+1}"
            ).add_to(m)

        st_folium(m, width=700)
        st.success("Zonas de manejo geradas!")
        st.markdown("📥 [Clique aqui para acessar a pasta no Google Drive](https://drive.google.com/drive/folders/0B0fNfM3PvQx3fk9LTFNOUThLMHM?usp=sharing) após a finalização da exportação.

⚠️ Certifique-se de que a pasta `gee_exports` esteja acessível na sua conta do Drive e que você tenha compartilhado ela ou configurado o acesso corretamente.")

        # Gerar relatório PDF
        media_ndvi = np.mean([x[0] for x in dados])
        media_ndre = np.mean([x[1] for x in dados])
        media_sbi = np.mean([x[2] for x in dados])
        media_alt = np.mean([x[3] for x in dados])

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Relatório de Zonas de Manejo", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Período: {data_inicio} a {data_fim}", ln=True)
        pdf.cell(200, 10, txt=f"Número de zonas: {k_zonas}", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Médias gerais das variáveis:", ln=True)
        pdf.cell(200, 10, txt=f"NDVI: {media_ndvi:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"NDRE: {media_ndre:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"SBI: {media_sbi:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"Altimetria: {media_alt:.2f} m", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                st.download_button("📄 Baixar Relatório PDF", f, file_name="relatorio_zonas_manejo.pdf")

elif not area_geom:
    st.warning("Envie uma área primeiro.")