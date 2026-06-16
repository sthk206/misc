"""
mem0 / Qdrant memory timeline viewer.

Shows memories in the order they were added, with:
  - a "memories added over time" chart
  - a filterable, time-sorted table
  - a cumulative growth line (watch the store fill up)

Run:
    pip install streamlit qdrant-client pandas
    streamlit run mem0_timeline.py

Then set the connection below to match your store.
"""

import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient

# ---- CONNECTION -----------------------------------------------------------
# LOCAL embedded file store (the .sqlite case): point at the FOLDER.
CLIENT_PATH = "/tmp/qdrant"          # <-- change to your Qdrant folder
# Or, if running a server instead, comment the line above and use:
# CLIENT_URL = "http://localhost:6333"

COLLECTION = "mem0"                   # <-- change if your collection differs
# ---------------------------------------------------------------------------


@st.cache_resource
def get_client():
    # Use path= for local file store, url= for a server.
    return QdrantClient(path=CLIENT_PATH)
    # return QdrantClient(url=CLIENT_URL)


@st.cache_data
def load_points():
    client = get_client()
    all_points = []
    offset = None
    while True:
        pts, offset = client.scroll(
            COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(pts)
        if offset is None:
            break

    rows = []
    for p in all_points:
        pl = p.payload or {}
        rows.append({
            "id": str(p.id),
            "memory": pl.get("data"),
            "user_id": pl.get("user_id"),
            "agent_id": pl.get("agent_id"),
            "run_id": pl.get("run_id"),
            "created_at": pl.get("created_at"),
            "updated_at": pl.get("updated_at"),
        })
    df = pd.DataFrame(rows)
    if "created_at" in df:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df.sort_values("created_at").reset_index(drop=True)


st.title("mem0 memory timeline")

df = load_points()
if df.empty:
    st.warning("No points found. Check CLIENT_PATH and COLLECTION.")
    st.stop()

# ---- Filters --------------------------------------------------------------
users = ["all"] + sorted([u for u in df["user_id"].dropna().unique()])
choice = st.selectbox("Filter by user_id", users)
view = df if choice == "all" else df[df["user_id"] == choice]

st.caption(f"{len(view)} memories"
           + (f"  ·  {view['created_at'].min()} → {view['created_at'].max()}"
              if view["created_at"].notna().any() else ""))

# ---- Added over time ------------------------------------------------------
if view["created_at"].notna().any():
    by_day = (view.dropna(subset=["created_at"])
                  .set_index("created_at")
                  .resample("D").size()
                  .rename("added"))
    st.subheader("Memories added per day")
    st.bar_chart(by_day)

    st.subheader("Cumulative total over time")
    st.line_chart(by_day.cumsum().rename("total"))
else:
    st.info("No usable 'created_at' timestamps in payloads — showing table only.")

# ---- Time-ordered table ---------------------------------------------------
st.subheader("Memories in order added")
st.dataframe(
    view[["created_at", "user_id", "memory", "id"]],
    use_container_width=True,
    hide_index=True,
)
