import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from collections import Counter
from io import BytesIO
from openpyxl.drawing.image import Image
from matplotlib.ticker import FuncFormatter
import tempfile
import re


from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import nltk
nltk.download('stopwords')


st.set_page_config(layout="wide")
st.title("Performance Business Analysis")

# ==========================================
# INPUT
# ==========================================
uploaded_chat = st.file_uploader(
    "📩 Upload file riwayat chat",
    type=['xlsx'],
)

uploaded_profit = st.file_uploader(
    "💰 Upload file profit",
    type=['xlsx'],
)

uploaded_traffic = st.file_uploader(
    "📈 Upload file traffic",
    type=['xlsx'],
)


# ==========================================
# STOPWORDS
# ==========================================
factory = StopWordRemoverFactory()
stop_factory = factory.get_stop_words()

nltk_stop = stopwords.words('indonesian')

custom_stop = [
    "kak","min","mba","mas","ya","yg","nya","dong",'yang','di','ke','dari',
    'untuk','dengan','atau','itu','ini','utk','kakak','saya','bisa','hai',
    "nih",'halo','hai','siang','pagi','malam','bos','kah','sy','yah','kaa',
    'aja','tapi','tp','cm','dlu','dulu','klo','yng'
]

all_stopwords = set(
    stop_factory +
    nltk_stop +
    custom_stop
)

# ==========================================
# NORMALISASI
# ==========================================
normalisasi = {
    "apakah": "apa","membeli": "beli","dibeli": "beli","pembelian": "beli","tutupnya": "tutup",
    "castem": "custom","skrg": "sekarang","nggaa": "tidak","brp": "berapa","dikirim": "kirim",
    "pengiriman": "kirim","mengirim": "kirim","bagnya": "bag","nggak": "tidak","gak": "tidak",
    "ga": "tidak","tdk": "tidak","tokonya": "toko", "ad": "ada","udah": "sudah","blm": "belum",
    "bsk": "besok","ofline": "offline","lihat2": "lihat","ofline store": "toko","Co": "checkout",
    'readay':'ready','adaa':'ada','mhon':'mohon','tlong':'tolong','mentidaknti':'lanjut','barangnya':'barang',
    "sdh":'sudah',"tidakn":"tidak","bs":"bisa","dikirim":"kirim","bls":"balas","sdkit":"sedikit",
    "ngtidak":"tidak"
}

# ==========================================
# CLEANING
# ==========================================
def clean_text(text):

    text = str(text).lower()

    for k,v in normalisasi.items():
        text = text.replace(k,v)

    import re

    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [w for w in words if w not in all_stopwords]

    return " ".join(words)

tab1, tab2, tab3, tab4 = st.tabs(
[
"📩 Chat Analysis",
"💰 Profit Analysis",
"📈 Traffic Analysis",
"📥 Summary and Export"
]
)


missing_files = []

if not uploaded_chat:
            missing_files.append("Chat")

if not uploaded_profit:
            missing_files.append("Profit")

if not uploaded_traffic:
            missing_files.append("Traffic")

if missing_files:
            st.warning(f"Silakan upload file: {', '.join(missing_files)}")   

# ==========================================
# CHAT
# ==========================================
with tab1:
    top_word = st.slider(
        "Top kata yang ditampilkan",
        5,
        50,
        20)
    top_produk = st.slider(
        "Top barang yang ditampilkan",
        5,
        50,
        20)    
    if uploaded_chat:

        chat_df = pd.read_excel(uploaded_chat)

        # tanggal chat
        chat_df['waktuchat'] = pd.to_datetime(
            chat_df['waktuchat']
        )

        # waktu dijawab
        chat_df['waktuchat_jawab'] = pd.to_datetime(
            chat_df['waktuchat_jawab']
        )

        # jam
        chat_df['jam_chat'] = chat_df['waktuchat'].dt.hour

        chat_df['jam_jawab'] = chat_df[
            'waktuchat_jawab'
        ].dt.hour

        # clean pesan
        chat_df['clean_pesan'] = chat_df['pesan'].apply(clean_text)

        st.success("Data berhasil diproses")

        # ===================================================
        # 1 WORD FREQUENCY
        # ===================================================

        rows = []

        for _, row in chat_df.iterrows():
            kata_list = str(row['clean_pesan']).split()
            kategori = row['kategori']

            for kata in kata_list:
                rows.append([kata, kategori])

        kata_df = pd.DataFrame(rows, columns=['Kata', 'Kategori'])

        # total frekuensi kata
        freq_kata = (
            kata_df.groupby('Kata')
            .size()
            .reset_index(name='Frekuensi')
        )

        # kategori yang paling sering untuk tiap kata
        kategori_terbanyak = (
            kata_df.groupby('Kata')['Kategori']
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )

        word_df = (
            freq_kata
            .merge(kategori_terbanyak, on='Kata')
            .sort_values(by='Frekuensi', ascending=False)
        )

        word_df = word_df.head(top_word)

        fig1, ax1 = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=word_df,
            x='Frekuensi',
            y='Kata',
            ax=ax1, errorbar=None
        )
        ax1.set_title("Top Kata yang Paling Sering Muncul")
        st.subheader("🔤 Kata Paling Sering Muncul")
        st.pyplot(fig1)

        selected_word = st.selectbox(
            "Pilih kata",
            word_df['Kata']
        )

        contoh_pesan = (
            chat_df[
                chat_df['clean_pesan'].str.contains(
                    selected_word,
                    case=False,
                    na=False
                )
            ][['nama','clean_pesan','kategori']]
            .head(top_word)
        )

        st.dataframe(contoh_pesan)

        # ===================================================
        # 2 PRODUK
        # ===================================================

        produk_df = (
            chat_df['produk']
            .value_counts()
            .reset_index()
        )

        produk_df.columns=['Produk','Jumlah']

        produk_df = produk_df.head(top_produk)

        fig2, ax2 = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=produk_df,
            x='Jumlah',
            y='Produk',
            ax=ax2
        )
        st.subheader("📝 Produk Paling Banyak Ditanyakan")
        ax2.set_xticks(range(0, produk_df['Jumlah'].max()+1))
        st.pyplot(fig2)
        selected_product = st.selectbox(
            "Pilih produk",
            produk_df['Produk']
        )

        detail_produk = (
            chat_df[
                chat_df['produk']==selected_product
            ][[
                'nama',
                'pesan',
                'checkout',
                'alasan tidak co'
            ]]
        )

        st.dataframe(detail_produk)
        # ===================================================
        # CHECKOUT ANALYSIS
        # ===================================================
        checkout_df = (
            chat_df['checkout']
            .value_counts()
            .reset_index()
        )

        checkout_df.columns=[
            'Checkout',
            'Jumlah'
        ]

        fig10, ax10 = plt.subplots()

        ax10.pie(
            checkout_df['Jumlah'],
            labels=checkout_df['Checkout'],
            autopct='%1.1f%%'
        )

        st.subheader("💰 Checkout Analysis")

        st.pyplot(fig10)
        # ===================================================
        # CHECKOUT ANALYSIS
        # ===================================================
        alasan_df = (
            chat_df['alasan tidak co']
            .value_counts()
            .reset_index()
        )

        alasan_df.columns=[
            'Alasan',
            'Jumlah'
        ]

        fig11, ax11 = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=alasan_df,
            x='Jumlah',
            y='Alasan',
            ax=ax11
        )

        st.subheader("❌ Alasan Tidak Checkout")

        st.pyplot(fig11)
        # ===================================================
        # JAM CHAT MASUK
        # ===================================================
        jam_chat = (
            chat_df['jam_chat']
            .value_counts()
            .sort_index()
        )

        fig3, ax3 = plt.subplots(figsize=(12,5))

        jam_chat.plot(
            kind='bar',
            ax=ax3
        )

        st.subheader("Jam Chat Masuk")

        st.pyplot(fig3)

        ramai_chat = jam_chat.idxmax()

        st.info(
            f"Jam paling ramai pelanggan menghubungi adalah sekitar pukul {ramai_chat}:00"
        )


        # ===================================================
        # JAM CS MENJAWAB
        # ===================================================
        jam_jawab = (
            chat_df['jam_jawab']
            .value_counts()
            .sort_index()
        )

        fig4, ax4 = plt.subplots(figsize=(12,5))

        jam_jawab.plot(
            kind='bar',
            ax=ax4
        )

        st.subheader("Jam CS Menjawab")

        st.pyplot(fig4)

        ramai_jawab = jam_jawab.idxmax()

        st.info(
            f"Jam CS paling aktif menjawab adalah sekitar pukul {ramai_jawab}:00"
        )
        # ===================================================
        # DURASI MENJAWAB
        # ===================================================
        st.subheader("⏱️ Rata-rata Waktu Balas CS")

        durasi_cols = [c for c in chat_df.columns if "durasi" in c.lower()]

        if durasi_cols:
            durasi_col = durasi_cols[0]

            chat_df[durasi_col] = pd.to_numeric(chat_df[durasi_col], errors="coerce")

            st.write(chat_df[durasi_col].describe())

            st.success(
                f"🔥 Rata-rata waktu balas CS: **{chat_df[durasi_col].mean():.1f} menit**"
            )
        else:
            st.info("Kolom durasi tidak ditemukan.")
        # ===================================================
        # TOP VIEW
        # ===================================================
with tab2:
    if uploaded_profit:
        profit_df = pd.read_excel(uploaded_profit)
        if "Halaman Produk Dilihat" in profit_df.columns:
            col_view = "Halaman Produk Dilihat"
        elif "Klik" in profit_df.columns:
            col_view = "Klik"
        if "Produk" in profit_df.columns:
            col_produk = "Produk"
        elif "Nama produk" in profit_df.columns:
            col_produk = "Nama produk"
        view_df = (
        profit_df
        .groupby(col_produk)[col_view]
        .sum()
        .reset_index()
        .sort_values(col_view, ascending=False)
        .head(top_produk)
        )
        fig5, ax5 = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=view_df,
            y=col_produk,
            x=col_view,
            ax=ax5
        )
        st.subheader("👀 Top Barang Dilihat")
        st.pyplot(fig5)

        st.dataframe(view_df)

        # ===================================================
        # KERANJANG
        # ===================================================
        if "Dimasukkan ke Keranjang (Produk)" in profit_df.columns:
            col_cart = "Dimasukkan ke Keranjang (Produk)"
        elif "Klik hingga Menambahkan Produk ke Keranjang" in profit_df.columns:
            col_cart = "Klik hingga Menambahkan Produk ke Keranjang"
        cart_df = (
        profit_df
        .groupby(col_produk)[col_cart]
        .sum()
        .reset_index()
        .sort_values(col_cart, ascending=False)
        .head(top_produk))
        fig6, ax6 = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=cart_df,
            y=col_produk,
            x=col_cart,
            ax=ax6
        )

        st.subheader("🛒 Top Barang Ditambahkan Keranjang")
        st.pyplot(fig6)
        st.dataframe(cart_df)
        # ===================================================
        # UANG
        # ===================================================
        if "Penjualan (Pesanan Siap Dikirim) (IDR)" in profit_df.columns:
            col_sales = "Penjualan (Pesanan Siap Dikirim) (IDR)"
        elif "GMV (Rp)" in profit_df.columns:
            col_sales = "GMV (Rp)"
    
        sales_df = (
        profit_df
        .groupby(col_produk)[col_sales]
        .sum()
        .reset_index()
        .sort_values(col_sales, ascending=False)
        .head(top_produk))
        fig7, ax7 = plt.subplots(figsize=(10,5))
        sns.barplot(
            data=sales_df,
            y=col_produk,
            x=col_sales,
            ax=ax7
        )
        ax7.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'Rp{x:,.0f}'))
        st.subheader("💵 Top Barang Terlaris (Uang)")
        st.pyplot(fig7)
        sales_df[col_sales] = sales_df[col_sales].map(lambda x: f"Rp{x:,.0f}")
        st.dataframe(sales_df)

        # ===================================================
        # KUANTITAS
        # ===================================================
        if "Produk (Pesanan Siap Dikirim)" in profit_df.columns:
            col_qty = "Produk (Pesanan Siap Dikirim)"
        elif "Pesanan SKU" in profit_df.columns:
            col_qty = "Pesanan SKU"
    
        qty_df = (
        profit_df
        .groupby(col_produk)[col_qty]
        .sum()
        .reset_index()
        .sort_values(col_qty, ascending=False)
        .head(top_produk))
        fig8, ax8 = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=qty_df,
            y=col_produk,
            x=col_qty,
            ax=ax8
        )

        st.subheader("🛎️ Top Barang Terlaris (Kuantitas)")
        st.pyplot(fig8)
        st.dataframe(qty_df)

with tab3:
        # ===================================================
        # PENGUNJUNG BARU
        # ===================================================
    if uploaded_traffic:
        traffic_df = pd.read_excel(uploaded_traffic)
        if "Pengunjung Baru" in traffic_df.columns:
            visitor_new = traffic_df["Pengunjung Baru"].sum()
        elif "Klik Unik" in traffic_df.columns:
            visitor_new = traffic_df["Klik Unik"].sum()
        st.metric(
        "🆕 Pengunjung Baru",
        f"{visitor_new:,.0f}")
        # ===================================================
        # PENGUNJUNG LAMA
        # ===================================================
        if "Pengunjung Lama" in traffic_df.columns:
            visitor_old = traffic_df["Pengunjung Lama"].sum()
        elif "Klik" in traffic_df.columns:
            visitor_old = (traffic_df["Klik"].sum()-traffic_df["Klik Unik"].sum())
        st.metric(
        "♻ Pengunjung Lama",
        f"{visitor_old:,.0f}")
        # ===================================================
        # TOTAL PENGUNJUNG
        # ===================================================
        if "Total Pengunjung" in traffic_df.columns:
            col_visitor = "Total Pengunjung"
        elif "Klik" in traffic_df.columns:
            col_visitor = "Klik"

        if "Tanggal" in traffic_df.columns:
            tanggal_col = "Tanggal"
            traffic_df[tanggal_col] = pd.to_datetime(
                traffic_df[tanggal_col],
                format='%d-%m-%Y',
                errors='coerce'
            )

        elif "Waktu" in traffic_df.columns:
            tanggal_col = "Waktu"
            traffic_df[tanggal_col] = pd.to_datetime(
                traffic_df[tanggal_col],
                format='%Y-%m-%d',
                errors='coerce'
            )

        # Jumlah pengunjung per hari
        traffic_daily = (
            traffic_df
            .groupby(tanggal_col)[col_visitor]
            .sum()
            .reset_index()
        )

        fig9, ax9 = plt.subplots(figsize=(12,5))

        sns.lineplot(
            data=traffic_daily,
            x=tanggal_col,
            y=col_visitor,
            marker='o',
            ax=ax9
        )

        st.metric(
            "👥 Total Pengunjung",
            f"{traffic_daily[col_visitor].sum():,.0f}"
        )
        ax9.set_xlabel("Tanggal")
        ax9.set_ylabel("Jumlah Pengunjung")

        st.subheader("📈 Total Pengunjung")
        st.pyplot(fig9)

        def save_chart(workbook,fig,sheet,pos="E2"):

                with tempfile.NamedTemporaryFile(
                        suffix=".png",
                        delete=False
                    ) as tmp:

                        fig.savefig(
                            tmp.name,
                            bbox_inches='tight'
                        )

                        img = Image(tmp.name)

                        workbook[sheet].add_image(
                            img,
                            pos
                        )
    output = None
    if uploaded_chat and uploaded_profit and uploaded_traffic:    
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:

                # ===================
                # Executive Summary
                # ===================

                summary_df = pd.DataFrame({
                    "Metrik":[
                        "Jumlah Chat",
                        "Jumlah Produk",
                        "Jumlah Kategori",
                        "Jam Chat Tersibuk",
                        "Jam CS Paling Aktif",
                        "Durasi Rata-Rata Jawab",
                        "Pengunjung Baru",
                        "Pengunjung Lama",
                        "Total Pengunjung"
                    ],

                    "Nilai":[
                        len(chat_df),
                        chat_df['produk'].nunique(),
                        chat_df['kategori'].nunique(),
                        f"{ramai_chat}:00",
                        f"{ramai_jawab}:00",
                        f"{chat_df[durasi_col].mean():.1f} menit",
                        f"{visitor_new:,.0f}",
                        f"{visitor_old:,.0f}",
                        f"{traffic_daily[col_visitor].sum():,.0f}"
                    ]
                })

                summary_df.to_excel(
                    writer,
                    sheet_name="Executive Summary",
                    index=False
                )


                # ===================
                # Top Words
                # ===================

                contoh_pesan.to_excel(
                    writer,
                    sheet_name="Top Words",
                    index=False
                )

                # ===================
                # Jam Chat
                # ===================

                jam_chat.reset_index().to_excel(
                    writer,
                    sheet_name="Jam Chat",
                    index=False
                )


                # ===================
                # Jam Jawab
                # ===================

                jam_jawab.reset_index().to_excel(
                    writer,
                    sheet_name="Jam Jawab",
                    index=False
                )
                # ===================
                # Produk Tanya
                # ===================
                detail_produk.to_excel(
                    writer,
                    sheet_name="Produk Tanya",
                    index=False
                )
                # ===================
                # Top View
                # ===================
                view_df.to_excel(
                    writer,
                    sheet_name="Top View",
                    index=False
                )
                # ===================
                # Keranjang
                # ===================
                cart_df.to_excel(
                    writer,
                    sheet_name="Keranjang",
                    index=False
                )
                # ===================
                # UANG
                # ===================
                sales_df.to_excel(
                    writer,
                    sheet_name="Terlaris (Uang)",
                    index=False
                )           
                # ===================
                # KUANTITAS
                # ===================
                qty_df.to_excel(
                    writer,
                    sheet_name="Terlaris (Kuantitas)",
                    index=False
                ) 
                # ===================
                # Total Pengunjung
                # ===================
                traffic_daily.to_excel(
                    writer,
                    sheet_name="Total Pengunjung",
                    index=False
                )
                # ===================
                # Checkout
                # ===================
                checkout_df.to_excel(
                    writer,
                    sheet_name="Checkout",
                    index=False
                )
                # ===================
                # Alasan Checkout
                # ===================
                alasan_df.to_excel(
                    writer,
                    sheet_name="Checkout",index=False
                    )         

        workbook = writer.book
        save_chart(workbook,fig1,"Top Words")
        save_chart(workbook,fig2,"Produk Tanya")
        save_chart(workbook,fig3,"Jam Chat")
        save_chart(workbook,fig4,"Jam Jawab")
        save_chart(workbook,fig5,"Top View")
        save_chart(workbook,fig6,"Keranjang")
        save_chart(workbook,fig7,"Terlaris (Uang)")
        save_chart(workbook,fig8,"Terlaris (Kuantitas)")
        save_chart(workbook,fig9,"Total Pengunjung")
        save_chart(workbook,fig10,"Checkout","E2")
        save_chart(workbook,fig11,"Checkout","E25")
        output.seek(0)
with tab4: 
     
        col1,col2,col3 = st.columns(3)

        with col1:

            if uploaded_chat:

                st.metric(
                    "Jumlah Chat",
                    len(chat_df)
                )
        with col2:

            if uploaded_traffic and 'traffic_daily' in locals():

                top_day = traffic_daily.loc[
                    traffic_daily[col_visitor].idxmax()
                ]

                st.metric(
                    "Hari Teramai",
                    top_day[tanggal_col].strftime("%d-%m-%Y"),
                    f"{top_day[col_visitor]:,.0f} pengunjung"
                )
        with col3:

            if uploaded_chat:

                st.metric(
                    "Peak Hour",
                    f"{ramai_chat}:00"
                    )
        nama_file_user = st.text_input("Nama file hasil unduhan:", value="hasil_analisis")

        if not nama_file_user.endswith(".xlsx"):
            nama_file_final = f"{nama_file_user}.xlsx"
        else:
            nama_file_final = nama_file_user
        if output is not None:
            st.download_button(
                label="📥 Download Hasil Analisis",
                data=output,
                file_name=nama_file_final,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
