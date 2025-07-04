# model/aboutme.py
import streamlit as st

def show_creator():
    st.header("👩‍💻 Kharisma Qaulam")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div style='text-align: justify;'>
            Saya adalah lulusan <b>Teknik Geodesi Universitas Diponegoro</b> yang sedang bertransisi
            menjadi <b>Data Analyst/Data Scientist</b>.
            Berpengalaman di <b>Customer Segmentation, A/B Testing, People Analytics</b>
            dengan Python, SQL, dan Power BI.
            <br><br>
            Saat ini mengikuti bootcamp intensif & akan magang bidang data bulan Juli.
            Misi saya: memberi dampak sosial lewat data dan membangun
            <b>personal branding</b> seputar career switching & buku.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("📬 Kontak & Profil")
        st.markdown("- 📧 kharisma.qaulam@gmail.com")
        st.markdown("- 💼 [LinkedIn](https://www.linkedin.com/in/kharismaqaulam/)")
        st.markdown("- 🐱 [GitHub](https://github.com/kharismqf)")
        st.markdown("- 📝 [Medium](https://medium.com/@qaulamk)")

    with col2:
        st.image("images/image6.jpeg", width=240, caption="Kharisma Qaulam")
