import subprocess
import torch
import gc  # For garbage collection
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from fastapi import FastAPI, HTTPException, Request
from pyngrok import ngrok
import uvicorn
import nest_asyncio
import time  # For adding delay

# Patch the existing event loop in Colab to allow running Uvicorn
nest_asyncio.apply()
os.system('ngrok authtoken <YOUR_NGROK_TOKEN>')

app = FastAPI()

# Global flag to track if the model is already loaded
model_loaded = False

# GPU management functions
def check_gpu_memory():
    try:
        gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        print("GPU Memory Info:\n", gpu_info)
    except subprocess.CalledProcessError as e:
        print(f"Error checking GPU memory: {e}")

def check_gpu_processes():
    try:
        gpu_processes = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        return gpu_processes
    except subprocess.CalledProcessError as e:
        return f"Error checking GPU processes: {e}"

# Using psutil for better process management
import psutil

def kill_gpu_processes():
    global model_loaded
    if model_loaded:
        print("Model is already loaded. Skipping process termination.")
        return "Model already loaded, skipping GPU process kill."

    try:
        gpu_processes = subprocess.check_output("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True).decode("utf-8")
        pids = [pid.strip() for pid in gpu_processes.split("\n") if pid.strip()]

        if not pids:
            return "No GPU processes found."

        for pid in pids:
            try:
                print(f"Killing process {pid}")
                if psutil.pid_exists(int(pid)):
                    proc = psutil.Process(int(pid))
                    proc.terminate()  # Try a graceful termination first
                    time.sleep(1)  # Give it a second to terminate gracefully

                    if proc.is_running():
                        proc.kill()  # Force kill if it's still running
                        print(f"Process {pid} was forcefully killed.")
                    else:
                        print(f"Process {pid} terminated successfully.")
                else:
                    print(f"Process {pid} does not exist anymore.")
            except psutil.NoSuchProcess:
                print(f"Process {pid} does not exist or was already terminated.")
            except subprocess.CalledProcessError as e:
                print(f"Error killing process {pid}: {e}")
        return "Killed all processes (or already terminated)."
    except subprocess.CalledProcessError as e:
        return f"Error fetching GPU processes: {e}"

# Model loading and cleanup
def load_model():
    global model, tokenizer, device, model_loaded
    model_path = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"

    if model_loaded:
        print("Model is already loaded. Skipping model reload.")
        return tokenizer, device

    if not os.path.exists(model_path):
        model_name = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"
    else:
        model_name = model_path

    # Check GPU memory before loading model
    print("Checking GPU memory before loading model...")
    check_gpu_memory()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    model_loaded = True  # Mark model as loaded
    return tokenizer, device

def cleanup():
    global model, model_loaded
    if not model_loaded:
        print("Model not loaded, skipping cleanup.")
        return

    print("Cleaning up model and freeing GPU memory...")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    model_loaded = False  # Mark model as unloaded

    # Delay to ensure GPU memory is fully released
    print("Waiting 10 seconds for GPU memory to be fully released...")
    time.sleep(10)

# Summarization function
def summarize_text(
        text: str,
        min_length: int = 100,
        max_length: int = 300,
        temperature: float = 0.5,
        num_beams: int = 5,
        repetition_penalty: float = 1.3,
        do_sample: bool = False
) -> str:
    """Summarizes the given text using the LLM model with controlled length."""

    input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length + 50,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=2,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            attention_mask=attention_mask,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.replace("لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: ", "").strip()

    return summary

# Ngrok setup
public_url = ngrok.connect(8000, bind_tls=False)  # Use HTTP
print(f"Public URL: {public_url}")

@app.post("/model-status")
async def model_status(request: Request):
    try:
        data = await request.json()
        action = data.get("action")

        if action == "load":
            tokenizer, device = load_model()
            return {"status": "Model loaded"}
        elif action == "cleanup":
            cleanup()
            return {"status": "Model cleaned up"}
        elif action == "gpu_check":
            gpu_info = check_gpu_processes()
            return {"gpu_info": gpu_info}
        elif action == "gpu_kill":
            kill_result = kill_gpu_processes()
            return {"gpu_kill": kill_result}
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/summarize")
async def summarize(request: Request):
    global model_loaded
    try:
        if not model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded. Please load the model first.")

        data = await request.json()
        text = data.get("text")
        min_length = data.get("min_length", 100)
        max_length = data.get("max_length", 300)
        temperature = data.get("temperature", 0.5)
        num_beams = data.get("num_beams", 5)
        repetition_penalty = data.get("repetition_penalty", 1.3)
        do_sample = data.get("do_sample", False)

        if not text:
            raise HTTPException(status_code=400, detail="The 'text' field is required.")

        summary = summarize_text(text, min_length, max_length, temperature, num_beams, repetition_penalty, do_sample)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        log_level="info",
        timeout_keep_alive=75
    )
    server = uvicorn.Server(config)
    server.run()
