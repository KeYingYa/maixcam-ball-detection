"""
MaixCam2 纯推流 —— 浏览器查看 + 录像 + 连拍训练
文档: https://wiki.sipeed.com/maixpy/doc/zh/video/webrtc_streaming.html
"""

from maix import app, camera, display, err, image, time, webrtc
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, os, shutil, threading

# ============================================================
# 配置
# ============================================================

STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
STREAM_FPS    = 30
STREAM_BITRATE = 3_000_000
STREAM_GOP    = 15

WEB_HTTP_PORT      = 8000
WEB_SIGNALING_PORT = 8001
STUN_SERVER        = "stun:stun.miwifi.com:3478"

LOCAL_RECORD_DIR        = "/root/maixcam2_recordings"
LOCAL_BURST_DIR          = "/root/maixcam2_burst"
LOCAL_RECORD_FREE_MARGIN = 100 * 1024 * 1024
RECORDING_WRITE_LOCK    = threading.Lock()

# ============================================================
# Web 页面
# ============================================================

WEB_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MaixCam2 图传</title>
<style>
html,body{margin:0;min-height:100%;background:#1a1a2e;color:#fff;font-family:Arial,"Microsoft YaHei",sans-serif}
body{display:flex;justify-content:center;align-items:center;box-sizing:border-box;padding:20px}
#container{width:100%;max-width:1200px;text-align:center}
#title{margin-bottom:16px;font-size:clamp(20px,3vw,32px);font-weight:bold;opacity:0.9}
#video-wrap{position:relative;overflow:hidden;width:100%;border-radius:16px;background:#000;box-shadow:0 12px 40px rgba(0,0,0,0.5)}
#video{display:block;width:100%;max-height:70vh;background:#000}
#status{position:absolute;bottom:14px;right:14px;padding:6px 12px;border-radius:10px;background:rgba(0,0,0,0.6);font-size:14px;backdrop-filter:blur(4px)}
.row{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:10px;margin:0 0 14px}
.btn{height:40px;border:1px solid rgba(255,255,255,0.15);border-radius:10px;color:#fff;font-size:15px;cursor:pointer;padding:0 16px;min-width:120px}
.btn:hover:not(:disabled){filter:brightness(1.15)}
.btn:disabled{cursor:not-allowed;opacity:0.45}
.btn-red{background:#c0392b}.btn-red.recording{background:#6b1a1a;box-shadow:0 0 0 3px rgba(192,57,43,0.3)}
.btn-green{background:#27ae60}.btn-green:hover:not(:disabled){background:#2ecc71}
.btn-orange{background:#d35400}.btn-orange.bursting{background:#27ae60;box-shadow:0 0 0 3px rgba(39,174,96,0.3)}
.btn-gray{background:#505050}
select{height:40px;border:1px solid rgba(255,255,255,0.15);border-radius:10px;color:#fff;font-size:15px;padding:0 10px;background:#505050;cursor:pointer}
.mono{font-family:Consolas,monospace;font-size:16px;min-width:50px}
.msg{flex-basis:100%;min-height:18px;color:#aaa;font-size:13px}
@media(max-width:600px){body{padding:10px}.btn{height:36px;font-size:13px;min-width:90px;padding:0 12px}select{height:36px;font-size:13px}}
</style>
</head>
<body>
<main id="container">
<div id="title">MaixCam2 图传</div>

<div class="row">
<label>录像保存</label>
<select id="record-target"><option value="computer">电脑端</option><option value="maixcam">MaixCam2</option></select>
<button class="btn btn-red" id="record-btn" disabled>● 开始录制</button>
<span class="mono" id="record-timer">00:00</span>
<div class="msg" id="record-msg">连接视频后可录制</div>
</div>

<div class="row">
<button class="btn btn-orange" id="burst-btn">⚡ 开始连拍(10张/秒)</button>
<span class="mono" id="burst-count">0 张</span>
<div class="msg" id="burst-msg">点击开始,直接下载到电脑</div>
</div>

<div id="video-wrap">
<video id="video" autoplay playsinline muted></video>
<div id="status">正在连接</div>
</div>
</main>
<script>
const signalingPort=__SIGNALING_PORT__;
const stunServer="__STUN_SERVER__";
const clientId=Math.random().toString(36).slice(2,12);
const ws=new WebSocket(`ws://${location.hostname}:${signalingPort}/${clientId}`);
const video=document.getElementById("video");
const statusEl=document.getElementById("status");

// --- WebRTC 连接 ---
let peer=null;
function setStatus(t){statusEl.textContent=t}
function waitIce(c){if(c.iceGatheringState==="complete")return Promise.resolve();return new Promise(r=>{const h=()=>{if(c.iceGatheringState==="complete"){c.removeEventListener("icegatheringstatechange",h);r()}};c.addEventListener("icegatheringstatechange",h);setTimeout(()=>{c.removeEventListener("icegatheringstatechange",h);r()},3000)})}
async function handleOffer(msg){
if(peer){stopRec();peer.close()}
peer=new RTCPeerConnection({bundlePolicy:"max-bundle",iceServers:[{urls:[stunServer]}]});
peer.ontrack=e=>{video.srcObject=e.streams[0];video.play();restoreRec();burstBtn.disabled=false;setStatus("已连接")};
peer.onconnectionstatechange=()=>{if(peer.connectionState==="failed"||peer.connectionState==="disconnected"||peer.connectionState==="closed"){stopRec();recordBtn.disabled=true;setStatus("连接断开")}};
await peer.setRemoteDescription({type:msg.type,sdp:msg.sdp});
await peer.setLocalDescription(await peer.createAnswer());
await waitIce(peer);
ws.send(JSON.stringify({id:"server",type:peer.localDescription.type,sdp:peer.localDescription.sdp}));
}
ws.onopen=()=>{setStatus("正在获取视频");ws.send(JSON.stringify({id:"server",type:"request"}))};
ws.onmessage=async e=>{if(typeof e.data!=="string")return;const m=JSON.parse(e.data);if(m.type==="offer"){try{await handleOffer(m)}catch(ex){console.error(ex);setStatus("连接失败")}}};
ws.onerror=()=>setStatus("信令连接失败");
ws.onclose=()=>{if(!peer||peer.connectionState!=="connected")setStatus("信令已断开")};

// --- 录像 ---
const recordTarget=document.getElementById("record-target");
const recordBtn=document.getElementById("record-btn");
const recordTimer=document.getElementById("record-timer");
const recordMsg=document.getElementById("record-msg");
let mediaRecorder=null,recordedChunks=[],recDest="computer",recStart=0,recTimerId=null;
function restoreRec(){const ok=video.srcObject&&video.srcObject.getVideoTracks().some(t=>t.readyState==="live");recordBtn.disabled=!ok||!("MediaRecorder"in window)||mediaRecorder!==null;recordBtn.classList.remove("recording");recordBtn.textContent="● 开始录制"}

async function finishRec(){
if(!mediaRecorder){restoreRec();return}
clearInterval(recTimerId);recTimerId=null;
recordBtn.disabled=true;recordBtn.classList.remove("recording");recordBtn.textContent="保存中...";
const mime=mediaRecorder.mimeType||"video/webm";
const ext=mime.toLowerCase().includes("mp4")?"mp4":"webm";
const t=((d=new Date)=>{const p=v=>String(v).padStart(2,"0");return d.getFullYear()+p(d.getMonth()+1)+p(d.getDate())+"_"+p(d.getHours())+p(d.getMinutes())+p(d.getSeconds())})();
const blob=new Blob(recordedChunks,{type:mime});recordedChunks=[];
if(!blob.size){recordMsg.textContent="录制失败:无数据";mediaRecorder=null;restoreRec();return}
try{
if(recDest==="maixcam"){
const r=await fetch("/api/recordings",{method:"POST",headers:{"Content-Type":blob.type,"X-Recording-Extension":ext,"X-Recording-Timestamp":t},body:blob});
const j=await r.json();
if(!r.ok||!j.ok)throw new Error(j.error||"HTTP "+r.status);
recordMsg.textContent="已保存到 MaixCam2: "+j.path;
}else{const u=URL.createObjectURL(blob);const a=document.createElement("a");a.href=u;a.download=t+"."+ext;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),30000);recordMsg.textContent="已保存: "+t+"."+ext}
}catch(ex){console.error(ex);recordMsg.textContent="保存失败: "+ex.message}
finally{mediaRecorder=null;restoreRec()}
}
function startRec(){
if(!video.srcObject||!("MediaRecorder"in window)){recordMsg.textContent="不支持录制";return}
try{
const mimes=["video/mp4;codecs=avc1.42E01E","video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm"];
let m="";for(const x of mimes){if(MediaRecorder.isTypeSupported&&!MediaRecorder.isTypeSupported(x))continue;m=x;break}
mediaRecorder=new MediaRecorder(video.srcObject,{mimeType:m,videoBitsPerSecond:__RECORD_BITRATE__});
recordedChunks=[];recDest=recordTarget.value;
mediaRecorder.ondataavailable=e=>{if(e.data&&e.data.size)recordedChunks.push(e.data)};
mediaRecorder.onstop=finishRec;
mediaRecorder.start(1000);
recStart=performance.now();recordTimer.textContent="00:00";
recTimerId=setInterval(()=>{const s=Math.floor((performance.now()-recStart)/1000);const m=Math.floor(s/60);const sec=s%60;recordTimer.textContent=(m>0?String(m).padStart(2,"0")+":":"")+String(sec).padStart(2,"0")},500);
recordBtn.classList.add("recording");recordBtn.textContent="■ 停止录制";
recordMsg.textContent=recDest==="maixcam"?"录制中,停止后保存到 MaixCam2":"录制中,停止后下载";
}catch(ex){console.error(ex);recordMsg.textContent="录制失败: "+ex.message;restoreRec()}
}
function stopRec(){if(mediaRecorder&&mediaRecorder.state==="recording"){recordBtn.disabled=true;recordBtn.textContent="正在停止...";mediaRecorder.stop()}}
recordBtn.onclick=()=>{if(mediaRecorder&&mediaRecorder.state==="recording")stopRec();else if(!mediaRecorder)startRec()};
window.addEventListener("beforeunload",e=>{if(mediaRecorder&&mediaRecorder.state==="recording"){e.preventDefault();e.returnValue=""}});

// --- 连拍训练 (10张/秒 → 本地文件夹) ---
const burstBtn=document.getElementById("burst-btn");
const burstCountEl=document.getElementById("burst-count");
const burstMsg=document.getElementById("burst-msg");
let burstHandle=null,burstCount=0,burstCanvas=null,burstCtx=null;
function downloadBlob(blob,name){
const u=URL.createObjectURL(blob);
const a=document.createElement("a");
a.href=u;a.download=name;
document.body.appendChild(a);a.click();a.remove();
setTimeout(()=>URL.revokeObjectURL(u),2000);
}
async function burstOne(){
if(!video.srcObject||!video.videoWidth)return;
const idx=burstCount;
try{
burstCanvas.width=video.videoWidth;burstCanvas.height=video.videoHeight;
burstCtx.drawImage(video,0,0);
const blob=await new Promise(r=>burstCanvas.toBlob(r,"image/jpeg",0.92));
if(!blob){burstMsg.textContent="截图失败";return}
const pad=n=>String(n).padStart(6,"0");
downloadBlob(blob,"train_"+pad(idx)+".jpg");
burstCount++;burstCountEl.textContent=burstCount+" 张";burstMsg.textContent="连拍中... "+burstCount+" 张";
}catch(ex){burstMsg.textContent="下载失败: "+ex.message}
}
burstBtn.onclick=()=>{
if(burstHandle){
clearInterval(burstHandle);burstHandle=null;
burstBtn.classList.remove("bursting");burstBtn.textContent="⚡ 开始连拍(10张/秒)";
burstMsg.textContent="已停止,共 "+burstCount+" 张,保存到下载文件夹";
}else{
burstCount=0;burstCountEl.textContent="0 张";
burstCanvas=burstCanvas||document.createElement("canvas");burstCtx=burstCanvas.getContext("2d");
burstMsg.textContent="连拍中,首次可能需要允许下载...";
burstHandle=setInterval(burstOne,100);
burstBtn.classList.add("bursting");burstBtn.textContent="⏹ 停止连拍";
}
};
</script>
</body>
</html>
"""

WEB_PAGE_BYTES = (
    WEB_PAGE.replace("__SIGNALING_PORT__", str(WEB_SIGNALING_PORT))
    .replace("__STUN_SERVER__", STUN_SERVER)
    .replace("__RECORD_BITRATE__", str(STREAM_BITRATE))
    .encode("utf-8")
)

# ============================================================
# HTTP 服务器
# ============================================================

class WebPageHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(WEB_PAGE_BYTES)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(WEB_PAGE_BYTES)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/recordings":
            self.send_json(404, {"ok": False, "error": "接口不存在"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self.send_json(400, {"ok": False, "error": "录像数据为空"})
            return
        extension = self.headers.get("X-Recording-Extension", "webm").lower()
        if extension not in ("mp4", "webm"):
            self.send_json(400, {"ok": False, "error": "不支持的录像格式"})
            return
        tmp = None
        try:
            os.makedirs(LOCAL_RECORD_DIR, exist_ok=True)
            with RECORDING_WRITE_LOCK:
                free = shutil.disk_usage(LOCAL_RECORD_DIR).free
                if content_length + LOCAL_RECORD_FREE_MARGIN > free:
                    self.send_json(507, {"ok": False, "error": "存储空间不足"})
                    return
                ts = self.headers.get("X-Recording-Timestamp", "")
                valid = (len(ts) == 15 and ts[8] == "_" and ts[:8].isdigit() and ts[9:].isdigit())
                timestamp = ts if valid else datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}.{extension}"
                final = os.path.join(LOCAL_RECORD_DIR, filename)
                seq = 1
                while os.path.exists(final):
                    filename = f"{timestamp}_{seq:02d}.{extension}"
                    final = os.path.join(LOCAL_RECORD_DIR, filename)
                    seq += 1
                tmp = f"{final}.part"
                remaining = content_length
                with open(tmp, "wb") as f:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ConnectionError("上传中断")
                        f.write(chunk)
                        remaining -= len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, final)
                tmp = None
            print(f"Recording saved: {final} ({content_length} bytes)")
            self.send_json(200, {"ok": True, "filename": filename, "path": final, "size": content_length})
        except Exception as exc:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            print(f"Failed to save recording: {exc}")
            self.send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format, *args):
        return


class WebPageServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_page_server():
    server = WebPageServer(("", WEB_HTTP_PORT), WebPageHandler)
    thread = threading.Thread(target=server.serve_forever, name="web-page", daemon=True)
    try:
        thread.start()
    except Exception:
        server.server_close()
        raise
    return server, thread


def stop_web_page_server(server, thread):
    if server is None:
        return
    try:
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2)
    except Exception as exc:
        print(f"Warning: failed to stop web page server: {exc}")


# ============================================================
# 主程序
# ============================================================

def main():
    # 双通道摄像头: NV21(编码) + RGB(本地预览)
    disp = display.Display()
    stream_cam = camera.Camera(STREAM_WIDTH, STREAM_HEIGHT, image.Format.FMT_YVU420SP, fps=STREAM_FPS)
    preview_cam = stream_cam.add_channel(disp.width(), disp.height(), image.Format.FMT_RGB888, fps=STREAM_FPS)

    # WebRTC 推流 (H.264 硬件编码)
    web_server = webrtc.WebRTC(
        stream_type=webrtc.WebRTCStreamType.WEBRTC_STREAM_H264,
        rc_type=webrtc.WebRTCRCType.WEBRTC_RC_CBR,
        bitrate=STREAM_BITRATE, gop=STREAM_GOP,
        signaling_port=WEB_SIGNALING_PORT, stun_server=STUN_SERVER,
        http_server=False,
    )
    err.check_raise(web_server.bind_camera(stream_cam))
    err.check_raise(web_server.start())

    # 网页服务器 (浏览器 UI)
    page_server, page_thread = start_web_page_server()

    urls = web_server.get_urls()
    print("=" * 50)
    print("WebRTC 推流已启动!")
    if urls:
        for url in urls:
            print(f"  {url}")
    else:
        print(f"  http://<设备IP>:{WEB_HTTP_PORT}")
    print("=" * 50)

    # 主循环: 本地预览
    report_ms = time.ticks_ms()
    report_cnt = 0

    while not app.need_exit():
        img = preview_cam.read()
        disp.show(img)
        report_cnt += 1

        now_ms = time.ticks_ms()
        if time.ticks_diff(now_ms, report_ms) >= 1000:
            fps = report_cnt * 1000.0 / time.ticks_diff(now_ms, report_ms)
            print(f"预览: {fps:.1f} fps")
            report_ms = now_ms
            report_cnt = 0

    # 清理
    stop_web_page_server(page_server, page_thread)
    web_server.stop()
    preview_cam.close()
    stream_cam.close()
    disp.close()


if __name__ == "__main__":
    main()
