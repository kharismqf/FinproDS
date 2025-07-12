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
            I'm currently enrolled in an intensive data bootcamp and will graduate in <b>July</b>. 
            My mission is to <b>create social impact through data</b> and <b>build a meaningful personal brand</b>
            by <b>deepening my knowledge</b> in data science and combining it with geospatial insights from my academic background.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("<h3 style='color:#0b2c4c;'>📬 Contact & Profile</h3>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="background-color: #FDF3E6; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
                <ul style="list-style-type: none; padding-left: 0; font-size: 16px;">
                    <li>📧 <b>Email:</b> <a href="mailto:qaulamk@gmail.com">qaulamk@gmail.com</a></li>
                    <li>💼 <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/kharismaqaulam/" target="_blank">kharismaqaulam</a></li>
                    <li>🐱 <b>GitHub:</b> <a href="https://github.com/kharismqf" target="_blank">kharismqf</a></li>
                    <li>📝 <b>Medium:</b> <a href="https://medium.com/@qaulamk" target="_blank">@qaulamk</a></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.image("images/image6.jpeg", width=240, caption="Kharisma Qaulam")
