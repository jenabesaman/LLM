from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from qa_app import process_question  # Import the function from qa_app.py
import uvicorn

app = FastAPI()


@app.post("/ask")
async def ask_question(request: Request):
    """API endpoint to answer questions based on the hardcoded context in qa_app."""
    try:
        # Parse the JSON body of the request
        data = await request.json()

        # Check if "question" is provided in the request body
        question_text = data.get("text")
        if not question_text:
            raise HTTPException(status_code=400, detail="The 'question' field is required.")

        # Call the function from qa_app.py to get the answer
        answer = process_question(question_text)

        # Return the answer
        if answer:
            return {"answer": answer}
        else:
            raise HTTPException(status_code=404, detail="No answer found.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
