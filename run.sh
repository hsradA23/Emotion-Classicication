#!/bin/bash
source bin/activate
uvicorn main:app --reload & cd webapp && npm run dev 