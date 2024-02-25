import time
import json
import base64
import asyncio

import os
from flask import render_template, request, Response, jsonify
from werkzeug.utils import redirect

from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play


from openai import OpenAI

def query(query, filename):
    client = OpenAI(api_key="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")
    query = "What happened in the last hour. Say everything that happened?"
    transcript = open(filename, 'r').read()
    completion = client.chat.completions.create(

    model="gpt-3.5-turbo",
    messages=[
                {"role": "system", "content": "You are a guard parsing through files of events to summarize, identify, and explain. Answer questions based off of files"},
                {"role": "user", "content": f"Transcript of all events in last 24 hours: {transcript}\nQuestion: {query}"}
            ]
        )
    return completion.choices[0].message.content

