from openai import OpenAI
import os
import streamlit as st

def summarize_observations(observations, taxon_name, model="gpt-4o"):
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])

    text_entries = []
    obs_text = "\n".join([f"{o['species']} at {o['location']} on {o['observed_on']}" for o in observations])
    prompt = f"Here are some recent wildlife observations:\n\n{obs_text}\n\nSummarize what was observed and any interesting patterns."

    # joined_text = "\n".join(text_entries)
    # prompt = (
    #     f"Summarize the most recent {len(text_entries)} observations of {taxon_name}. "
    #     f"Highlight interesting trends or notable facts in under 100 words:\n\n{joined_text}"
    # )

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages
    )

    return response.choices[0].message.content
