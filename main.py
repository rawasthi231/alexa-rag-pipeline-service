import os
import tempfile
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from rag_pipeline import GeminiRAGPipeline


rag_pipeline = GeminiRAGPipeline()
rag_pipeline.setup_collection()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_file_from_url(url, target_folder):
    try:
        print("Downloading file from URL: " + url)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get filename from URL or generate one
        filename = os.path.basename(url) or 'downloaded_file'
        print("Filename: " + filename)
        if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            filename += '.txt'
            
        filepath = os.path.join(target_folder, secure_filename(filename))
        print("Filepath: " + filepath)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return filepath
    except Exception as e:
        raise Exception(f"Error downloading file from URL: {str(e)}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# Simple root route to help platform port detection
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "service": "alexa-rag-pipeline-service"})


@app.route("/ingest", methods=["POST"])
def ingest_documents():
    # Check if the post request has the file part
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                rag_pipeline.ingest_documents([filepath])
                return jsonify({
                    "status": "success", 
                    "message": "Document uploaded and ingested successfully",
                    "filename": filename
                })
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
            finally:
                # Clean up the uploaded file after processing
                try:
                    os.remove(filepath)
                except:
                    pass
        else:
            return jsonify({
                "status": "error", 
                "message": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            })
    
    # Handle URL-based ingestion
    elif request.is_json:
        data = request.get_json()
        file_urls = data.get("file_urls", [])
        file_paths = data.get("file_paths", [])
        print("File URLs: " + str(file_urls))
        print("File Paths: " + str(file_paths))
        if not file_urls and not file_paths:
            return jsonify({"status": "error", "message": "No files or URLs provided"})
            
        temp_files = []
        try:
            # Download files from URLs
            for url in file_urls:
                print("URL: " + url)
                try:
                    filepath = download_file_from_url(url, app.config['UPLOAD_FOLDER'])
                    print("Filepath: " + filepath)
                    file_paths.append(filepath)
                    temp_files.append(filepath)
                except Exception as e:
                    return jsonify({
                        "status": "error", 
                        "message": f"Failed to download file from {url}: {str(e)}"
                    })
            
            # Process all files
            if file_paths:
                rag_pipeline.ingest_documents(file_paths)
                return jsonify({
                    "status": "success", 
                    "message": f"Successfully ingested {len(file_paths)} document(s)"
                })
                
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
            
        finally:
            # Clean up temporary files
            for filepath in temp_files:
                try:
                    os.remove(filepath)
                except:
                    pass
    
    return jsonify({"status": "error", "message": "Invalid request"})


@app.route("/query", methods=["POST"])
def query_documents():
    data = request.json
    job_id = data.get("job_id", "")
    task = data.get("task", "")
    print(task)
    if not task:
        return jsonify({"status": "error", "message": "No task provided"})

    try:
        result = rag_pipeline.query(task)
        print(result)
        # result = "This is a test response"
        return jsonify({"status": "success", "result": result, "job_id": job_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5500)
