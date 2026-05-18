import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from xgboost import XGBRegressor
import requests
from PIL import Image
from io import BytesIO


warnings.filterwarnings("ignore")

# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="Football Player Performance Evaluation and Value Prediction",
    page_icon="",
    layout="wide"
)

# ===================================
# CUSTOM CSS (SOCCER UI)
# ===================================

st.markdown("""
<style>

.stApp{
background-image: url("https://img95.699pic.com/photo/50402/6632.jpg_wh300.jpg!/fh/300/quality/90");
background-size: cover;
background-attachment: fixed;
}

.main{
background: rgba(0,0,0,0.65);
padding: 2rem;
border-radius:20px;
}

h1,h2,h3{
color:white;
}

section[data-testid="stSidebar"]{
background: linear-gradient(180deg,#07111f,#0b2239);
}

div[data-testid="metric-container"]{
background: rgba(255,255,255,0.08);
border:1px solid rgba(255,255,255,0.2);
padding:20px;
border-radius:20px;
backdrop-filter: blur(12px);
box-shadow:0 8px 32px rgba(0,0,0,0.35);
}

.player-card{
background: rgba(255,255,255,0.08);
padding:30px;
border-radius:25px;
backdrop-filter: blur(12px);
box-shadow:0px 8px 30px rgba(0,0,0,0.4);
margin-bottom:25px;
}

.big-title{
font-size:40px;
font-weight:800;
color:white;
text-align:center;
}

.sub{
text-align:center;
color:#9ad8ff;
font-size:20px;
}

.scout-badge{
padding:15px;
border-radius:15px;
font-weight:bold;
text-align:center;
font-size:22px;
}

.undervalued{
background:#0f5132;
color:#8fffb5;
}

.overvalued{
background:#842029;
color:#ffb3b3;
}

</style>
""", unsafe_allow_html=True)

# ===================================
# LOAD DATA
# ===================================

@st.cache_data
def load_data():
    app_df = pd.read_csv("data/app_dataset.csv")
    model_df = pd.read_csv("processed_playeriq_data.csv")
    return app_df, model_df


@st.cache_resource
def load_models():
    model = joblib.load("player_value_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


app_df, df_model = load_data()
model, scaler = load_models()

# ===================================
# FEATURES
# ===================================

features = [
'overall','potential','pace','shooting',
'passing','dribbling','defending',
'physic','age','height_cm','weight_kg'
]

# ===================================
# HERO SECTION
# ===================================

st.markdown(
"""
<div class='big-title'>
Football Player Performance Evaluation and Value Prediction
</div>

<div class='sub'>
A Powered Football Valuation & Talent Discovery Platform
</div>

<br>
""",
unsafe_allow_html=True
)

# ===================================
# SIDEBAR
# ===================================

# =========================
# SIDEBAR STYLING
# =========================
st.sidebar.markdown("""
<style>
.sidebar-title{
    font-size:20px;
    font-weight:800;
    color:white;
    margin-bottom:10px;
}

.filter-box{
    background:rgba(255,255,255,0.05);
    padding:12px;
    border-radius:12px;
    margin-bottom:15px;
    border:1px solid rgba(255,255,255,0.08);
}

.player-select{
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.sidebar.markdown('<div class="sidebar-title">⚽ Scouting Controls</div>', unsafe_allow_html=True)

# =========================
# FILTERS (CARD STYLE)
# =========================
st.sidebar.markdown('<div class="filter-box">', unsafe_allow_html=True)
min_rating = st.sidebar.slider("🎯 Minimum Overall", 50, 95, 70)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="filter-box">', unsafe_allow_html=True)
max_age = st.sidebar.slider("🧒 Maximum Age", 16, 40, 30)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# =========================
# FILTER LOGIC
# =========================
filtered = df_model[
    (df_model["overall"] >= min_rating) &
    (df_model["age"] <= max_age)
].copy()

if filtered.empty:
    st.warning("No players found")
    st.stop()

# Merge for names
merged = filtered.merge(app_df, left_index=True, right_index=True)

# =========================
# PLAYER SEARCH (ENHANCED)
# =========================
st.sidebar.markdown('<div class="filter-box">', unsafe_allow_html=True)

player_name = st.sidebar.selectbox(
    "🔍 Search Player",
    sorted(merged["long_name"].dropna().unique())
)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# =========================
# PLAYER INDEX
# =========================
player_idx = merged[merged["long_name"] == player_name].index[0]
model_row = filtered.loc[player_idx]

# =========================
# QUICK SUMMARY (NEW 🔥)
# =========================
st.sidebar.markdown('<div class="filter-box">', unsafe_allow_html=True)

st.sidebar.markdown(f"""
<b style="color:white;">Selected Player</b><br>
<span style="color:#9ad8ff;">{player_name}</span><br><br>

<b style="color:white;">Overall:</b> {int(model_row['overall'])} ⭐<br>
<b style="color:white;">Age:</b> {int(model_row['age'])}
""", unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)
# ===================================
# PREDICTION
# ===================================

X = model_row[features].values.reshape(1,-1)
X_scaled = scaler.transform(X)

pred = float(model.predict(X_scaled)[0])
actual = float(model_row["value_eur"])
diff = pred - actual

# ===================================
# PLAYER IMAGE + DETAILS (FIXED LAYOUT)
# ===================================

col_img, col_info = st.columns([1,2])

# -----------------------------
# LEFT COLUMN (PROFESSIONAL SCOUT CARD)
# -----------------------------
with col_img:

    st.markdown("###  Player Profile")

    # Stylish container
    st.markdown("""
    <style>

    .player-img{
        width: 260px;
        height: 260px;
        object-fit: cover;
        border-radius: 18px;
        border: 3px solid rgba(0,212,255,0.6);
        box-shadow: 0 0 25px rgba(0,212,255,0.35);
        transition: transform 0.3s ease;
    }

    .player-img:hover{
        transform: scale(1.03);
    }

    .img-caption{
        margin-top: 10px;
        color: #9ad8ff;
        font-size: 13px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="player-card-img">', unsafe_allow_html=True)

    if "player_face_url" in merged.columns:

        img_url = str(merged.loc[player_idx, "player_face_url"]).strip()

        try:
            response = requests.get(img_url, timeout=10)

            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, width=260, use_container_width=False)
            else:
                st.image(
                    "https://cdn-icons-png.flaticon.com/512/194/194938.png",
                    width=200
                )

        except:
            st.image(
                "https://cdn-icons-png.flaticon.com/512/194/194938.png",
                width=200
            )

    else:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/194/194938.png",
            width=200
        )

    st.markdown('<div class="img-caption">Official Player Profile Image</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RIGHT COLUMN (ALL DETAILS)
# -----------------------------
with col_info:

    # ==============================
    # GLOBAL STYLE (CLEAN + CONSISTENT)
    # ==============================
    st.markdown("""
    <style>
    .player-header{
        font-size:32px;
        font-weight:900;
        color:white;
        margin-bottom:5px;
    }

    .player-sub{
        color:#9ad8ff;
        font-size:14px;
        margin-bottom:20px;
    }

    .card{
        background: rgba(255,255,255,0.08);
        padding:18px;
        border-radius:16px;
        text-align:center;
        border:1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }

    .card:hover{
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }

    .label{
        font-size:11px;
        letter-spacing:1px;
        color:#9ad8ff;
        margin-bottom:6px;
    }

    .value{
        font-size:22px;
        font-weight:800;
        color:white;
    }

    .divider{
        height:1px;
        background: rgba(255,255,255,0.15);
        margin:20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==============================
    # HEADER
    # ==============================
    st.markdown(f"""
    <div class="player-header">{player_name}</div>
    <div class="player-sub"> Scouting Profile</div>
    """, unsafe_allow_html=True)

    # ==============================
    # TOP ROW (KEY STATS)
    # ==============================
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="label">AGE</div>
            <div class="value">{model_row['age']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="label">OVERALL</div>
            <div class="value">{model_row['overall']} </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="label">POTENTIAL</div>
            <div class="value">{model_row['potential']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ==============================
    # SECOND ROW (PHYSICAL + VALUE)
    # ==============================
    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="label">HEIGHT</div>
            <div class="value">{model_row['height_cm']} cm</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="card">
            <div class="label">WEIGHT</div>
            <div class="value">{model_row['weight_kg']} kg</div>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class="card">
            <div class="label">MARKET VALUE</div>
            <div class="value">{int(model_row['value_eur']):,}</div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================
    # SCOUT STATUS BADGE
    # ==============================
    st.markdown("<br>", unsafe_allow_html=True)

    if model_row["overall"] >= 85:
        st.markdown("""
        <div style="
            background:linear-gradient(90deg,#00c6ff,#0072ff);
            padding:12px;
            border-radius:12px;
            text-align:center;
            font-weight:700;
            color:white;
        ">
         Elite World-Class Player
        </div>
        """, unsafe_allow_html=True)

    elif model_row["overall"] >= 75:
        st.markdown("""
        <div style="
            background:linear-gradient(90deg,#00b09b,#96c93d);
            padding:12px;
            border-radius:12px;
            text-align:center;
            font-weight:700;
            color:white;
        ">
         Strong Professional Player
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background:linear-gradient(90deg,#f7971e,#ffd200);
            padding:12px;
            border-radius:12px;
            text-align:center;
            font-weight:700;
            color:black;
        ">
         Developing Talent
        </div>
        """, unsafe_allow_html=True)
# ===================================
# METRICS
# ===================================

# =========================
# METRICS (SLEEK COMPACT UI)
# =========================

st.markdown("""
<style>
.metric-mini{
    padding:12px 16px;
    border-bottom:1px solid rgba(255,255,255,0.15);
}

.metric-label{
    font-size:11px;
    color:#9ca3af;
    letter-spacing:1px;
}

.metric-value{
    font-size:20px;
    font-weight:700;
    color:white;
    margin-top:2px;
}

.metric-diff-pos{
    color:#22c55e;
}

.metric-diff-neg{
    color:#ef4444;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# --- Actual ---
with col1:
    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-label">ACTUAL</div>
        <div class="metric-value">€{actual:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Predicted ---
with col2:
    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-label">PREDICTED</div>
        <div class="metric-value">€{pred:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Difference ---
with col3:

    diff_class = "metric-diff-pos" if diff > 0 else "metric-diff-neg"
    arrow = "+" if diff > 0 else "-"

    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-label">DIFFERENCE</div>
        <div class="metric-value {diff_class}">
            {arrow}€{abs(diff):,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)
# ===================================
# RECOMMENDATION
# ===================================

# =========================
# SCOUT ALERT (PRO BANNER)
# =========================

st.markdown("""
<style>
.alert-box{
    padding:14px 18px;
    border-radius:10px;
    font-weight:600;
    display:flex;
    align-items:center;
    justify-content:space-between;
    border-left:5px solid;
    font-size:14px;
}

.alert-text{
    letter-spacing:0.5px;
}

.alert-positive{
    background:#052e1f;
    color:#22c55e;
    border-color:#22c55e;
}

.alert-negative{
    background:#3b0a0a;
    color:#ef4444;
    border-color:#ef4444;
}

.alert-tag{
    font-size:11px;
    padding:4px 8px;
    border-radius:6px;
    background:rgba(255,255,255,0.08);
    color:#9ca3af;
}
</style>
""", unsafe_allow_html=True)


if diff > 0:
    st.markdown(f"""
    <div class="alert-box alert-positive">
        <div class="alert-text">
            ✅ SCOUT ALERT: Undervalued Transfer Target
        </div>
        <div class="alert-tag">
            VALUE +€{abs(diff):,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="alert-box alert-negative">
        <div class="alert-text">
            ⚠ SCOUT ALERT: Overvalued Player
        </div>
        <div class="alert-tag">
            VALUE -€{abs(diff):,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===================================
# ATTRIBUTES
# ===================================

# =========================
# PLAYER ATTRIBUTES (ELITE UI)
# =========================

st.markdown("""
<style>

.attr-title{
    font-size:18px;
    font-weight:700;
    color:white;
    margin-bottom:15px;
}

.attr-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:14px;
}

.attr-card{
    background:#111827;
    padding:14px;
    border-radius:10px;
    border:1px solid #1f2937;
}

.attr-head{
    display:flex;
    justify-content:space-between;
    font-size:12px;
    color:#9ca3af;
    margin-bottom:6px;
}

.attr-value{
    font-weight:700;
    color:white;
}

.attr-bar{
    height:6px;
    border-radius:6px;
    background:#1f2937;
    overflow:hidden;
}

.attr-fill{
    height:100%;
    border-radius:6px;
    background:linear-gradient(90deg,#3b82f6,#22c55e);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="attr-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="attr-title">📊 Player Attributes</div>', unsafe_allow_html=True)

attributes = [
    ("⚡ Pace", model_row["pace"]),
    ("🎯 Shooting", model_row["shooting"]),
    ("🎮 Passing", model_row["passing"]),
    ("🕺 Dribbling", model_row["dribbling"]),
    ("🛡 Defending", model_row["defending"]),
    ("💪 Physical", model_row["physic"]),
]

st.markdown('<div class="attr-grid">', unsafe_allow_html=True)

for name, value in attributes:
    st.markdown(f"""
    <div class="attr-card">
        <div class="attr-head">
            <span>{name}</span>
            <span class="attr-value">{value}</span>
        </div>
        <div class="attr-bar">
            <div class="attr-fill" style="width:{value}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===================================
# TOP BARGAINS
# ===================================

st.markdown("## Top Undervalued Bargains")

sample = filtered.sample(min(15,len(filtered)),random_state=42).copy()

X_all = scaler.transform(sample[features])
sample["predicted"] = model.predict(X_all)
sample["diff"] = sample["predicted"] - sample["value_eur"]

top = sample.sort_values("diff",ascending=False).head(10)

st.dataframe(top[[c for c in ["long_name","overall","age","value_eur","predicted","diff"] if c in top.columns]])

# ===================================
# FOOTER
# ===================================

st.markdown("""
<br><br>
<hr style='border:1px solid rgba(255,255,255,0.2)'>

<center style='color:white'>
PlayerIQ . A Football Recruitment Platform
</center>
""",unsafe_allow_html=True)