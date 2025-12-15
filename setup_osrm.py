import os
import subprocess
import sys
import time

# --- CẤU HÌNH ---
# Link tải bản đồ (Mặc định là Việt Nam từ Geofabrik)
MAP_URL = "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf"
# Tên thư mục chứa dữ liệu
DATA_DIR = "osrm-data"
# Docker image
DOCKER_IMAGE = "osrm/osrm-backend:latest"
# Profile (car, bicycle, foot)
PROFILE = "car"
# Thuật toán (mld hoặc ch). MLD linh hoạt hơn.
ALGORITHM = "mld"

def run_command(cmd):
    """Chạy lệnh shell và kiểm tra lỗi"""
    print(f"Executing: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"❌ Lỗi khi chạy lệnh: {cmd}")
        sys.exit(1)

def download_file(url, folder):
    """Tải file bản đồ nếu chưa tồn tại"""
    filename = url.split('/')[-1]
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        print(f"✅ File {filename} đã tồn tại. Bỏ qua tải xuống.")
        return filename
    
    print(f"⬇️ Đang tải {filename} từ {url}...")
    # Sử dụng curl hoặc wget có sẵn trên hệ thống để tải cho nhanh
    # Nếu Windows không có curl, có thể dùng thư viện urllib của Python
    if sys.platform == "win32":
        import urllib.request
        urllib.request.urlretrieve(url, filepath)
    else:
        run_command(f"curl -L {url} -o {filepath}")
    
    print("✅ Tải xuống hoàn tất.")
    return filename

def main():
    # 1. Tạo thư mục dữ liệu
    abs_data_path = os.path.abspath(DATA_DIR)
    if not os.path.exists(abs_data_path):
        os.makedirs(abs_data_path)
        print(f"📁 Đã tạo thư mục: {abs_data_path}")

    # 2. Tải bản đồ
    map_filename = download_file(MAP_URL, abs_data_path)
    base_name = map_filename.replace(".osm.pbf", "") # tên file không đuôi

    # Đường dẫn file bên trong container (luôn là /data/...)
    docker_map_path = f"/data/{map_filename}"
    docker_osrm_path = f"/data/{base_name}.osrm"

    print("\n🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU (Có thể mất vài phút)...")

    # 3. Bước 1: Extract
    print("\n=== BƯỚC 1/3: EXTRACT ===")
    cmd_extract = (
        f"docker run -t -v \"{abs_data_path}:/data\" {DOCKER_IMAGE} "
        f"osrm-extract -p /opt/{PROFILE}.lua {docker_map_path}"
    )
    run_command(cmd_extract)

    # 4. Bước 2: Partition
    print("\n=== BƯỚC 2/3: PARTITION ===")
    cmd_partition = (
        f"docker run -t -v \"{abs_data_path}:/data\" {DOCKER_IMAGE} "
        f"osrm-partition {docker_osrm_path}"
    )
    run_command(cmd_partition)

    # 5. Bước 3: Customize
    print("\n=== BƯỚC 3/3: CUSTOMIZE ===")
    cmd_customize = (
        f"docker run -t -v \"{abs_data_path}:/data\" {DOCKER_IMAGE} "
        f"osrm-customize {docker_osrm_path}"
    )
    run_command(cmd_customize)

    # 6. Hướng dẫn chạy server
    print("\n✅ XỬ LÝ HOÀN TẤT!")
    print("-------------------------------------------------------")
    print("Để khởi chạy server, hãy copy và chạy lệnh dưới đây trong Terminal/CMD:")
    print("-------------------------------------------------------")
    
    run_server_cmd = (
        f"docker run -d -p 5000:5000 --name osrm-server --restart always "
        f"-v \"{abs_data_path}:/data\" {DOCKER_IMAGE} "
        f"osrm-routed --algorithm {ALGORITHM} {docker_osrm_path}"
    )
    
    print(run_server_cmd)
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()