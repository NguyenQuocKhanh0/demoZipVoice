import os
import subprocess
import requests

def run_cmd(cmd):
    print(f"🔹 Chạy lệnh: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại: {cmd}")

def download_with_token(url, dest_path, token):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Đã tải: {dest_path}")

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("Thiếu biến môi trường HF_TOKEN. Hãy đặt bằng: export HF_TOKEN=your_token")

    # Đăng nhập Hugging Face CLI
    run_cmd(f"huggingface-cli login --token {token}")

    # Tạo thư mục chứa model
    os.makedirs("zipvoice_finetune", exist_ok=True)

    # Danh sách file cần tải
    files = {
        "iter-525000-avg-2.pt": "https://huggingface.co/datasets/meandyou200175/temp_file/resolve/main/zip/epoch-46-all-speak-600h-en-norm.pt",
        "model.json": "https://huggingface.co/datasets/meandyou200175/temp_file/resolve/main/zip/model.json",
        "tokens.txt": "https://huggingface.co/datasets/meandyou200175/temp_file/resolve/main/zip/tokens.txt",
    }

    for filename, url in files.items():
        dest = os.path.join("zipvoice_finetune", filename)
        download_with_token(url, dest, token)

    # Cài đặt requirements
    if os.path.exists("requirements.txt"):
        run_cmd("pip install -r requirements.txt")
    else:
        print("⚠️ Không tìm thấy requirements.txt")

    print("\n🎉 Hoàn tất setup!")

if __name__ == "__main__":
    main()
