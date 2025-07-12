# model/aboutme.py
import streamlit as st

def show_creator():
    st.header("👩‍💻 Kharisma Qaulam")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div style='
                background-color: #FDF3E6;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                color: #000;
                font-size: 16px;
                line-height: 1.6;
                text-align: justify;
            '>
            Hello! I'm a <b>Geodetic Engineering graduate</b> from Universitas Diponegoro,
            currently transitioning into a career in <b>Data Analytics & Data Science</b>.
            <br><br>
            I have hands-on experience in <b>Customer Segmentation, A/B Testing, and People Analytics</b>
            using tools like <b>Python, SQL, and Power BI</b>.
            <br><br>
            I'm currently enrolled in an intensive data bootcamp and will be starting a data internship in <b>July</b>.
            My mission is to <b>create social impact through data</b> and build a strong
            <b>personal brand</b> around career switching and book insights.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("📬 Kontak & Profil")
        st.markdown("- 📧 qaulamk@gmail.com")
        st.markdown("- 💼 [LinkedIn](https://www.linkedin.com/in/kharismaqaulam/)")
        st.markdown("- 🐱 [GitHub](https://github.com/kharismqf)")
        st.markdown("- 📝 [Medium](https://medium.com/@qaulamk)")

    with col2:
        st.image("images/image6.jpeg", width=240, caption="Kharisma Qaulam")
