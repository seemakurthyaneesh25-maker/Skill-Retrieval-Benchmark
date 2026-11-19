import re, math
from collections import Counter
import numpy as np, pandas as pd, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Skill Retrieval Benchmark & Naming Optimizer", layout="wide")
SAMPLE_SKILLS = [
{"skill_id":"SK001","canonical_name":"PDF Summarizer","description":"Extract key information from PDF documents and produce concise summaries.","aliases":"document summary; pdf analysis","tags":"pdf;documents;summary","category":"Documents"},
{"skill_id":"SK002","canonical_name":"Email Drafter","description":"Draft, rewrite and polish professional emails using the requested tone and context.","aliases":"email writer; mail composition","tags":"email;writing;communication","category":"Writing"},
{"skill_id":"SK003","canonical_name":"Spreadsheet Analyzer","description":"Analyze spreadsheet data, identify trends, calculate statistics and create charts.","aliases":"excel analysis; csv analyzer","tags":"excel;csv;analytics;charts","category":"Data"},
{"skill_id":"SK004","canonical_name":"Meeting Minutes Generator","description":"Convert meeting notes or transcripts into minutes, decisions and action items.","aliases":"meeting summary; action item extractor","tags":"meeting;minutes;transcript","category":"Productivity"},
{"skill_id":"SK005","canonical_name":"Code Debugger","description":"Inspect source code, identify errors, explain causes and recommend fixes.","aliases":"debugging assistant; error fixer","tags":"programming;debugging;software","category":"Development"},
{"skill_id":"SK006","canonical_name":"Travel Planner","description":"Create travel itineraries based on destination, dates, budget and preferences.","aliases":"trip organizer; itinerary builder","tags":"travel;itinerary;planning","category":"Planning"},
{"skill_id":"SK007","canonical_name":"Image Caption Generator","description":"Describe image contents and generate concise captions or accessible alt text.","aliases":"photo describer; image description","tags":"image;caption;accessibility","category":"Media"},
{"skill_id":"SK008","canonical_name":"Resume Optimizer","description":"Improve resumes for clarity, role relevance and applicant tracking system keywords.","aliases":"CV improver; resume reviewer","tags":"resume;career;job application","category":"Career"}]
SAMPLE_QUERIES = [
{"query":"make a short overview of this research pdf","relevant_ids":"SK001"},
{"query":"write a polite message asking my professor for leave","relevant_ids":"SK002"},
{"query":"show trends and a graph from my csv file","relevant_ids":"SK003"},
{"query":"extract decisions and tasks from a call transcript","relevant_ids":"SK004"},
{"query":"why is my python program throwing an error","relevant_ids":"SK005"},
{"query":"organize a low cost three day holiday","relevant_ids":"SK006"},
{"query":"tell me what is in this photo for alt text","relevant_ids":"SK007"},
{"query":"make my CV better for a software job","relevant_ids":"SK008"}]
def tokens(x): return re.findall(r"[a-z0-9]+", str(x).lower())
def skill_doc(s): return " ".join(str(s.get(k,"")) for k in ["canonical_name","description","aliases","tags","category"])
def bm25(query, docs):
    ds=[tokens(d) for d in docs]; n=len(ds); avg=sum(map(len,ds))/max(n,1)
    df=Counter(t for d in ds for t in set(d)); out=[]
    for d in ds:
        tf=Counter(d); score=0.0
        for t in tokens(query):
            if t in df:
                idf=math.log(1+(n-df[t]+.5)/(df[t]+.5))
                score += idf*tf[t]*2.5/(tf[t]+1.5*(.25+.75*len(d)/max(avg,1)))
        out.append(score)
    return np.array(out)
def retrieve(query, skills, method):
    docs=[skill_doc(s) for s in skills]
    b=bm25(query,docs)
    vec=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
    mat=vec.fit_transform(docs+[query])
    dense=cosine_similarity(mat[-1],mat[:-1]).ravel()
    if method=="BM25": scores=b
    elif method=="Dense (TF-IDF proxy)": scores=dense
    else:
        scores=np.zeros(len(skills))
        for arr in (b,dense):
            for rank,idx in enumerate(np.argsort(-arr),1): scores[idx]+=1/(60+rank)
    return [(skills[i],float(scores[i]),float(b[i]),float(dense[i])) for i in np.argsort(-scores)]
def candidates(skill):
    parts=skill.get("canonical_name","Skill").split()
    tags=[x.strip() for x in skill.get("tags","").split(";") if x.strip()]
    first=parts[0] if parts else "Skill"
    rest=" ".join(parts[1:]) if len(parts)>1 else (tags[0].title() if tags else "Assistant")
    return list(dict.fromkeys([f"{rest} {first}",f"{first} {rest} Assistant",f"{tags[0].title()} {rest}" if tags else f"{first} Assistant"]))
def dcg(vals): return sum((2**v-1)/math.log2(i+2) for i,v in enumerate(vals))
def evaluate(ranked, relevant):
    rel=[3 if s["skill_id"] in relevant else 0 for s,_,_,_ in ranked]
    first=next((i+1 for i,v in enumerate(rel) if v>0),None)
    return (len({s["skill_id"] for s,_,_,_ in ranked[:5]} & relevant)/max(len(relevant),1),
            1/first if first and first<=10 else 0,
            dcg(rel[:10])/max(dcg(sorted(rel,reverse=True)[:10]),1e-9))

st.title("Skill Retrieval Benchmark and Naming Optimizer")
st.caption("Working prototype following the uploaded report: skill repository → retrieval → name suggestions → benchmark evaluation.")
with st.sidebar:
    method=st.selectbox("Retrieval method",["Hybrid (BM25 + Dense)","BM25","Dense (TF-IDF proxy)"])
    skills_file=st.file_uploader("Upload skills CSV",type=["csv"])
    queries_file=st.file_uploader("Upload benchmark queries CSV",type=["csv"])
skills_df=pd.read_csv(skills_file).fillna("") if skills_file else pd.DataFrame(SAMPLE_SKILLS)
queries_df=pd.read_csv(queries_file).fillna("") if queries_file else pd.DataFrame(SAMPLE_QUERIES)
skills=skills_df.to_dict("records")
tab1,tab2,tab3,tab4=st.tabs(["Live Search","Naming Optimizer","Benchmark","Skill Catalogue"])
with tab1:
    q=st.text_input("Enter a user request","Can you summarize this PDF report?")
    if q.strip():
        st.caption("Hybrid combines BM25 and TF-IDF cosine rankings using Reciprocal Rank Fusion (k=60).")
        for pos,(s,score,b,d) in enumerate(retrieve(q,skills,method)[:5],1):
            with st.container(border=True):
                left,right=st.columns([5,1])
                left.markdown(f"**{pos}. {s.get('canonical_name','')}** · `{s.get('skill_id','')}`")
                left.write(s.get("description",""))
                left.caption(f"Aliases: {s.get('aliases','—')} | Tags: {s.get('tags','—')}")
                right.metric("Rank score",f"{score:.4f}")
                right.caption(f"BM25 {b:.3f} · Dense {d:.3f}")
with tab2:
    ids=[str(s["skill_id"]) for s in skills]
    sid=st.selectbox("Select a skill",ids)
    selected=next(s for s in skills if str(s["skill_id"])==sid)
    st.write("**Current name:**",selected.get("canonical_name",""))
    st.write(selected.get("description",""))
    opts=candidates(selected)
    st.write("**Suggested names**")
    for i,name in enumerate(opts,1): st.write(f"{i}. {name}")
    preview=st.selectbox("Preview candidate",opts)
    st.success("Preview display name: "+preview)
    st.warning("Human approval is required. Check clarity, uniqueness and semantic fidelity. The stable skill_id must not change.")
with tab3:
    st.write("Metrics below are computed from the included synthetic demo labels. They are not validated research findings.")
    rows=[]
    for _,qr in queries_df.iterrows():
        rel=set(x.strip() for x in str(qr.get("relevant_ids","")).split(";") if x.strip())
        if rel:
            r=evaluate(retrieve(str(qr["query"]),skills,method),rel)
            rows.append({"Query":qr["query"],"Recall@5":r[0],"MRR@10":r[1],"nDCG@10":r[2]})
    result=pd.DataFrame(rows)
    if not result.empty:
        a,b,c=st.columns(3)
        a.metric("Mean Recall@5",f"{result['Recall@5'].mean():.3f}")
        b.metric("Mean MRR@10",f"{result['MRR@10'].mean():.3f}")
        c.metric("Mean nDCG@10",f"{result['nDCG@10'].mean():.3f}")
        st.dataframe(result,use_container_width=True)
        st.bar_chart(result.set_index("Query"))
with tab4:
    st.dataframe(skills_df,use_container_width=True,hide_index=True)
    st.download_button("Download skills CSV",skills_df.to_csv(index=False),"sample_skills.csv","text/csv")
    st.download_button("Download benchmark queries CSV",queries_df.to_csv(index=False),"sample_queries.csv","text/csv")
st.divider()
st.markdown("**Run locally:** `pip install -r requirements.txt` then `streamlit run app.py`. Replace sample records and relevance labels with reviewed project data before reporting research results.")
