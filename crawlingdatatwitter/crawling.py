import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import streamlit as st
from PIL import Image

def main():
    # Sidebar
    activity = ["Hasil Crawling Twitter", "Word Cloud", "Top Words"]
    choice = st.sidebar.selectbox("Menu", activity)

    # Muat gambar di awal
    image = Image.open('crawlingdatatwitter/emas.jpg')
    st.sidebar.image(image, caption='Indonesia Emas', use_column_width=True)  # Gambar di sidebar

    # Path file CSV sudah diketahui
    file_path = 'crawlingdatatwitter/hasil_prosesing.csv'
    
    # Baca file CSV
    data = pd.read_csv(file_path)

    # Menampilkan konten berdasarkan pilihan di sidebar
    if choice == "Hasil Crawling Twitter":
        st.title("Hasil Crawling Twitter dengan kata kunci Indonesia Emas")
        st.write(data.head(50))  # Menampilkan data dari file CSV
    
    elif choice == "Word Cloud":
        st.title("Word Cloud dari Hasil Crawling")
        generate_wordcloud(data)  # Menampilkan word cloud
    
    elif choice == "Top Words":
        st.title("Kata yang Paling Sering Muncul")
        plot_top_words(data)  # Menampilkan top words dalam bar chart

# Fungsi untuk menghasilkan word cloud
def generate_wordcloud(data):
    st.write("Menampilkan ke Wordcloud dan Stopword")
    
    data_text = ' '.join(data['stemming'].astype(str).tolist())  # Convert the column to string explicitly
    
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 'co', 'RT', '...', 'amp'])

    wc = WordCloud(stopwords=stopwords, background_color='black', max_words=500, width=800, height=400).generate(data_text)
    
    # Buat figure untuk plot matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')  # Sembunyikan axis
    st.pyplot(fig)  # Tampilkan figure di Streamlit

# Fungsi untuk menampilkan bar chart dari kata yang sering muncul
def plot_top_words(data):
    st.write("Kata yang Paling Sering Muncul")

    # Gabungkan semua teks dari kolom 'full_text'
    text = ' '.join(data['full_text'].astype(str).tolist())
    words = text.split()

    # Hitung frekuensi kata
    word_counts = Counter(words)

    # Ambil 12 kata yang paling sering muncul
    top_words = word_counts.most_common(12)

    words, counts = zip(*top_words)

    # Buat bar chart menggunakan matplotlib
    colors = plt.cm.Paired(range(len(words)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(words, counts, color=colors)

    ax.set_xlabel('Kata')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Kata yang sering muncul')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45)

    # Tambahkan label frekuensi di atas setiap bar
    for bar, num in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')

    # Tampilkan bar chart di Streamlit
    st.pyplot(fig)

if __name__ == '__main__':
    main()
