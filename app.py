import streamlit as st
from persona_agent.agent import PersonaAdaptiveAgent

st.set_page_config(page_title="Support Agent", page_icon="💬", layout="wide")

# Only use HTML for things that MUST be custom — nav + chat bubbles
# Everything else uses native Streamlit widgets
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0f1117; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* metric cards */
div[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 16px !important;
}
div[data-testid="metric-container"] label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.06em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 28px !important; font-weight: 700 !important; }
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* progress bars */
div[data-testid="stProgress"] > div > div { background: #1e2130 !important; }

/* text input */
div[data-testid="stTextInput"] input {
    background: #1e2130 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}
div[data-testid="stTextInput"] label { display: none !important; }
div[data-testid="stForm"] { border: none !important; background: transparent !important; padding: 0 !important; }

/* buttons */
.stButton > button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    width: 100% !important;
}
.stButton > button:hover { background: #4f46e5 !important; }

/* divider */
hr { border-color: #2d3148 !important; }

/* section headers override */
h3 { color: #94a3b8 !important; font-size: 11px !important; font-weight: 600 !important;
     text-transform: uppercase !important; letter-spacing: 0.08em !important; }
</style>
""", unsafe_allow_html=True)

# ── session state ──
if "agent"    not in st.session_state: st.session_state.agent    = PersonaAdaptiveAgent()
if "messages" not in st.session_state: st.session_state.messages = []
if "stats"    not in st.session_state:
    st.session_state.stats = {
        "total": 0, "escalations": 0, "resolved": 0,
        "counts": {"technical_expert":0,"frustrated_user":0,"business_executive":0,"general_user":0},
        "log": [],
    }

PERSONA_INFO = {
    "technical_expert":   {"label":"Technical",  "color":"#6366f1","emoji":"⚙️"},
    "frustrated_user":    {"label":"Frustrated",  "color":"#ef4444","emoji":"😤"},
    "business_executive": {"label":"Executive",   "color":"#f59e0b","emoji":"💼"},
    "general_user":       {"label":"General",     "color":"#10b981","emoji":"👤"},
}

# ── top nav (HTML ok here — it's simple and static) ──
st.markdown("""
<div style="background:#161922;border:1px solid #2d3148;border-radius:14px;
            padding:14px 22px;display:flex;align-items:center;
            justify-content:space-between;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:36px;height:36px;
                background:linear-gradient(135deg,#6366f1,#8b5cf6);
                border-radius:10px;display:flex;align-items:center;
                justify-content:center;color:white;font-weight:700;font-size:16px;">S</div>
    <div>
      <div style="color:#f1f5f9;font-size:15px;font-weight:600;">Support Agent</div>
      <div style="color:#64748b;font-size:12px;">Persona-Adaptive AI · Assignment Demo</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;color:#10b981;font-size:13px;font-weight:500;">
    <div style="width:8px;height:8px;background:#10b981;border-radius:50%;
                box-shadow:0 0 0 3px rgba(16,185,129,0.25);"></div>
    Agent online
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

# ════════════════════════════
#  LEFT — CHAT
# ════════════════════════════
with left:
    # chat window
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="height:380px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;gap:14px;text-align:center;">
              <div style="font-size:48px;">💬</div>
              <div style="color:#e2e8f0;font-size:17px;font-weight:600;">Start a conversation</div>
              <div style="color:#64748b;font-size:13px;line-height:1.8;">
                Type a message below and watch the agent<br>detect your persona and adapt its response.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    # user bubble — right aligned
                    c1, c2 = st.columns([1, 4])
                    with c2:
                        st.markdown(f"""
                        <div style="background:#6366f1;color:white;padding:12px 16px;
                                    border-radius:18px 18px 4px 18px;font-size:14px;
                                    line-height:1.6;margin:4px 0;">{msg['text']}</div>
                        """, unsafe_allow_html=True)
                else:
                    meta  = msg.get("meta", {})
                    p     = meta.get("persona", "general_user")
                    info  = PERSONA_INFO.get(p, PERSONA_INFO["general_user"])
                    conf  = int(meta.get("confidence", 0) * 100)
                    frust = meta.get("frustration", 0)
                    kb    = meta.get("kb_title", "")
                    esc   = meta.get("escalated", False)
                    text  = msg["text"].replace("\n\n", "<br><br>").replace("\n", "<br>")

                    # agent bubble — left aligned
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"""
                        <div style="background:#161922;border:1px solid #2d3148;color:#e2e8f0;
                                    padding:14px 16px;border-radius:4px 18px 18px 18px;
                                    font-size:14px;line-height:1.7;margin:4px 0;">{text}</div>
                        """, unsafe_allow_html=True)

                        # persona strip
                        frust_color = "#ef4444" if frust >= 7 else "#94a3b8"
                        kb_part = f"&nbsp;&nbsp;·&nbsp;&nbsp;📄 {kb}" if kb else ""
                        esc_part = "&nbsp;&nbsp;·&nbsp;&nbsp;🚨 <span style='color:#ef4444;font-weight:600;'>Escalated</span>" if esc else ""
                        st.markdown(f"""
                        <div style="background:#0f1117;border:1px solid #2d3148;border-radius:8px;
                                    padding:8px 14px;margin:4px 0 12px 0;font-size:12px;color:#64748b;
                                    display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
                          <span style="background:#1e1b4b;color:{info['color']};padding:2px 9px;
                                       border-radius:20px;font-weight:600;font-size:11px;">
                            {info['emoji']} {info['label']}
                          </span>
                          &nbsp;·&nbsp; Confidence <strong style="color:#94a3b8;">{conf}%</strong>
                          &nbsp;·&nbsp; Frustration <strong style="color:{frust_color};">{frust}/10</strong>
                          {kb_part}{esc_part}
                        </div>
                        """, unsafe_allow_html=True)

                    if esc and meta.get("handoff"):
                        st.markdown(f"""
                        <div style="background:#1c1505;border-left:3px solid #f59e0b;
                                    border-radius:8px;padding:12px 16px;margin:0 0 12px 0;
                                    font-size:11.5px;color:#fbbf24;font-family:monospace;
                                    white-space:pre-wrap;line-height:1.7;">{meta['handoff']}</div>
                        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # input
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input("msg", placeholder="Type your support question...",
                                       label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if st.button("↺  New conversation", use_container_width=True):
        st.session_state.agent    = PersonaAdaptiveAgent()
        st.session_state.messages = []
        st.rerun()

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "text": user_input})
        result = st.session_state.agent.handle_message(user_input)
        meta = {
            "persona":    result.persona_result.persona,
            "confidence": result.persona_result.confidence,
            "frustration":result.persona_result.frustration_score,
            "kb_title":   result.kb_article.title if result.kb_article else None,
            "escalated":  result.escalate,
            "handoff":    result.handoff_context,
        }
        st.session_state.messages.append({"role":"agent","text":result.response,"meta":meta})
        s = st.session_state.stats
        s["total"] += 1
        s["counts"][result.persona_result.persona] += 1
        s["escalations"] += int(result.escalate)
        s["resolved"]    += int(not result.escalate)
        s["log"].insert(0, {"persona": result.persona_result.persona,
                            "text": user_input, "escalated": result.escalate})
        s["log"] = s["log"][:6]
        st.rerun()

# ════════════════════════════
#  RIGHT — DASHBOARD
# ════════════════════════════
with right:
    s   = st.session_state.stats
    tot = max(s["total"], 1)

    st.markdown("### Live Dashboard")

    # stat cards using native st.metric
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Messages",    s["total"],       delta="this session")
        st.metric("Resolved",    s["resolved"],    delta=f"{int(s['resolved']/tot*100)}% rate" if s["total"] else "0%")
    with c2:
        st.metric("Escalations", s["escalations"], delta=f"{int(s['escalations']/tot*100)}% rate" if s["total"] else "0%",
                  delta_color="inverse")
        seen = len(set(m["meta"]["persona"] for m in st.session_state.messages if m["role"]=="agent")) if st.session_state.messages else 0
        st.metric("Personas seen", seen, delta="of 4 types")

    st.markdown("---")
    st.markdown("### Persona Breakdown")

    for key, info in PERSONA_INFO.items():
        count = s["counts"][key]
        pct   = count / tot if s["total"] else 0
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<span style='color:{info['color']};font-size:13px;font-weight:500;'>{info['emoji']} {info['label']}</span>", unsafe_allow_html=True)
            st.progress(pct)
        with col2:
            st.markdown(f"<div style='color:#64748b;font-size:12px;padding-top:6px;'>{count} · {int(pct*100)}%</div>", unsafe_allow_html=True)

    if s["log"]:
        st.markdown("---")
        st.markdown("### Recent Messages")
        for item in s["log"][:5]:
            info = PERSONA_INFO.get(item["persona"], PERSONA_INFO["general_user"])
            esc  = " 🚨" if item["escalated"] else ""
            preview = item["text"][:55] + ("…" if len(item["text"]) > 55 else "")
            st.markdown(f"""
            <div style="background:#161922;border:1px solid #2d3148;border-radius:10px;
                        padding:10px 13px;margin-bottom:8px;">
              <div style="margin-bottom:4px;">
                <span style="background:#1e2130;color:{info['color']};font-size:10px;
                             font-weight:700;padding:2px 8px;border-radius:6px;
                             text-transform:uppercase;letter-spacing:0.04em;">
                  {info['emoji']} {info['label']}{esc}
                </span>
              </div>
              <div style="font-size:12px;color:#64748b;">"{preview}"</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Try These")
    examples = [
        ("⚙️","Technical", "Getting a 429 on the API endpoint after 100 requests"),
        ("😤","Frustrated","THIS IS THE THIRD TIME IT BROKE!! UNACCEPTABLE!!!"),
        ("💼","Executive", "We need SLA details for our 200-seat enterprise contract"),
        ("👤","General",   "How do I reset my password?"),
        ("🚨","Escalate",  "I just want to talk to a human agent please"),
    ]
    for emoji, label, text in examples:
        st.markdown(f"""
        <div style="background:#161922;border:1px solid #2d3148;border-radius:10px;
                    padding:10px 13px;margin-bottom:7px;">
          <div style="font-size:10px;color:#64748b;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">
            {emoji} {label}
          </div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.4;">"{text}"</div>
        </div>
        """, unsafe_allow_html=True)
