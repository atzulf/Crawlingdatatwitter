# Nama 	: Ataka Dzulfikar
# NIM  	: 22537141002
# Prodi	: Teknologi Informasi / I

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    # Sidebar
    activity = ["Hasil Crawling Twitter", "Word Cloud", "Top Words"]
    choice = st.sidebar.selectbox("Menu", activity)

    # Muat gambar di awal
    image = Image.open('emas.jpg')
    st.sidebar.image(image, caption='Indonesia Emas', use_column_width=True)  # Gambar di sidebar

    # Path file CSV hasil crawling dan hasil labeling
    file_crawling = 'hasil_prosesing.csv'
    file_labeling = 'stemmingwithlabel.csv'
    
    # Baca file CSV hasil crawling dan hasil labeling
    data_crawling = pd.read_csv(file_crawling)
    data_labeling = pd.read_csv(file_labeling)

    # Menampilkan konten berdasarkan pilihan di sidebar
    if choice == "Hasil Crawling Twitter":
        st.title("Hasil Crawling Twitter dengan kata kunci Indonesia Emas")
        st.write("Disini saya berhasil mendapatkan 360 data dari Twitter:")
        st.write(data_crawling.head(361))  # Menampilkan data hasil crawling

        # Tambahkan tabel hasil labeling di bawah hasil crawling
        st.write("Berikut adalah tabel hasil labeling:")
        st.write(data_labeling[['stemming', 'label']].head(50))  # Menampilkan data hasil labeling

        # Proses Naive Bayes untuk analisis sentimen
        st.subheader("Analisis Sentimen menggunakan Naive Bayes")
        st.write("Distribusi label dalam data:")
        st.write(data_labeling['label'].value_counts())  # Cek distribusi label

        # Vectorizer dan model Naive Bayes
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data_labeling['stemming'])  # Proses teks dari kolom 'stemming'
        y = data_labeling['label']

        # Split data untuk melatih dan menguji model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)  # Latih model pada data train
        
        # Evaluasi model
        y_pred = model.predict(X_test)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Input teks dari pengguna
        input_text = st.text_area("Masukkan teks untuk analisis sentimen:")
        
        if st.button("Analisis Sentimen"):
            if input_text:
                input_vector = vectorizer.transform([input_text])
                prediction = model.predict(input_vector)[0]

                
                if prediction == "positive":
                    st.success("Sentimen: Positif")
                elif prediction == "neutral":
                    st.warning("Sentimen: Netral")
                else:
                    st.error("Sentimen: Negatif")
            else:
                st.error("Harap masukkan teks!")

    elif choice == "Word Cloud":
        st.title("Word Cloud dari Hasil Crawling")
        generate_wordcloud(data_crawling)  # Menampilkan word cloud
    
    elif choice == "Top Words":
        st.title("Kata yang Paling Sering Muncul")
        plot_top_words(data_crawling)  # Menampilkan top words dalam bar chart

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
