from __future__ import annotations

import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model" / "epoch_2000000.ckpt.ts"
SONG = Path(os.environ.get("NIGHTJAR_SOURCE_WAV", ROOT / "data" / "nightjar.wav")).expanduser()
AUDIO = ROOT / "audio"
HTML = ROOT / "index.html"

SR = 44_100
WINDOW_SEC = 20.0
HOP_SEC = 3.0
SEGMENT_SEC = 3.0
CROSSFADE_SEC = 0.35
TARGET_RMS = 0.10
SEED = 43
MIN_VALID_SEC = 8.0

VARIANTS = (
    ("near_previous", "Near previous anchor", 0.25, 0.00, "#2878bd"),
    ("near_next", "Near next anchor", 0.75, 0.00, "#36a852"),
    ("pingpong_curve", "Ping-pong near anchors + curve", None, 0.08, "#e5484d"),
)


def norm(y: np.ndarray, rms: float = TARGET_RMS, peak: float = 0.98) -> np.ndarray:
    y = np.nan_to_num(np.asarray(y, np.float32))
    if not len(y):
        return y
    y = y - float(y.mean())
    level = float(np.sqrt(np.mean(y * y)) + 1e-12)
    if level > 1e-10:
        y *= rms / level
    top = float(np.max(np.abs(y)) + 1e-12)
    if top > peak:
        y *= peak / top
    return y.astype(np.float32)


def pad(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, np.float32)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def tensor(y: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(y, np.float32)).view(1, 1, -1).to(device)


def array(x: torch.Tensor) -> np.ndarray:
    return np.nan_to_num(np.squeeze(x.detach().cpu().numpy())).astype(np.float32)


def load_model(device: str):
    rave = torch.jit.load(str(MODEL), map_location=device).eval()
    assert int(getattr(rave, "sr", SR)) == SR
    assert int(getattr(rave, "latent_size", 8)) == 8
    assert int(getattr(rave, "full_latent_size", 128)) == 128
    return rave


@torch.no_grad()
def encode(rave, y: np.ndarray, seed: int, device: str) -> np.ndarray:
    torch.manual_seed(seed)
    z = rave.encode(tensor(y, device))
    z = z[0] if isinstance(z, (tuple, list)) else z
    z = array(z)
    return (z[0] if z.ndim == 3 else z[None, :] if z.ndim == 1 else z).astype(np.float32)


@torch.no_grad()
def decode(rave, z: np.ndarray, device: str) -> np.ndarray:
    full = np.zeros((128, z.shape[1]), np.float32)
    full[:8] = z[:8]
    y = rave.decode(torch.from_numpy(full[None]).float().to(device))
    y = y[0] if isinstance(y, (tuple, list)) else y
    return array(y)


def summary(z: np.ndarray) -> np.ndarray:
    return z.mean(1).astype(np.float32)


def mix(z0: np.ndarray, z1: np.ndarray, t: float) -> np.ndarray:
    n = min(z0.shape[-1], z1.shape[-1])
    return ((1 - t) * z0[..., :n] + t * z1[..., :n]).astype(np.float32)


def curved(z0: np.ndarray, z1: np.ndarray, t: float, amount: float, seed: int) -> np.ndarray:
    base = mix(z0, z1, t)
    n = base.shape[-1]
    direction = z1[..., :n] - z0[..., :n]
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise -= (np.sum(noise * direction) / (np.sum(direction * direction) + 1e-12)) * direction
    noise /= np.sqrt(np.mean(noise * noise)) + 1e-12
    return (base + amount * np.sqrt(np.mean(direction * direction)) * noise).astype(np.float32)


def segment(rave, z: np.ndarray, device: str) -> np.ndarray:
    y = decode(rave, z, device)
    n = int(SEGMENT_SEC * SR)
    start = max(0, (len(y) - n) // 2)
    return norm(pad(y[start : start + n], n))


def join(parts: list[np.ndarray]) -> np.ndarray:
    fade = int(CROSSFADE_SEC * SR)
    out = parts[0]
    for part in parts[1:]:
        n = min(fade, len(out), len(part))
        t = np.linspace(0, 1, n, dtype=np.float32)
        cross = out[-n:] * np.cos(t * np.pi / 2) + part[:n] * np.sin(t * np.pi / 2)
        out = np.concatenate([out[:-n], cross, part[n:]]).astype(np.float32)
    return norm(out)


def audio_stats(y: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(y * y)) + 1e-12),
        "peak": float(np.max(np.abs(y)) + 1e-12),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)[0])),
    }


def render() -> dict:
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing RAVE model: {MODEL}")
    if not SONG.exists():
        raise FileNotFoundError(
            "Missing source song. Set NIGHTJAR_SOURCE_WAV=/path/to/nightjar.wav "
            "or place it at audio_generation/rave_midpoints/data/nightjar.wav."
        )
    AUDIO.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rave = load_model(device)
    song, _ = librosa.load(SONG, sr=SR, mono=True)
    window = int(WINDOW_SEC * SR)
    hop = int(HOP_SEC * SR)
    max_start = max(0, int((len(song) / SR - MIN_VALID_SEC) * SR))
    windows = [norm(pad(song[start : start + window], window)) for start in range(0, max_start + 1, hop)]
    anchors = [encode(rave, y, SEED + i, device) for i, y in enumerate(windows)]

    scaler = StandardScaler().fit(np.stack([summary(z) for z in anchors]))
    pca = PCA(n_components=3, random_state=SEED).fit(scaler.transform(np.stack([summary(z) for z in anchors])))
    anchor_xyz = pca.transform(scaler.transform(np.stack([summary(z) for z in anchors])))

    payload = {
        "meta": {"sr": SR, "hopSec": HOP_SEC, "segmentSec": SEGMENT_SEC, "crossfadeSec": CROSSFADE_SEC},
        "anchors": [
            {"i": i, "x": float(x), "y": float(y), "z": float(z), "time": i * HOP_SEC}
            for i, (x, y, z) in enumerate(anchor_xyz)
        ],
        "variants": [],
    }

    for name, label, fixed_t, curve, color in VARIANTS:
        parts, route = [], []
        for i in range(len(anchors) - 1):
            t = (0.25 if i % 2 == 0 else 0.75) if fixed_t is None else fixed_t
            z = curved(anchors[i], anchors[i + 1], t, curve, SEED + 10_000 + i) if curve else mix(anchors[i], anchors[i + 1], t)
            y = segment(rave, z, device)
            parts.append(y)
            x, y3, z3 = pca.transform(scaler.transform(summary(z)[None]))[0]
            route.append(
                {
                    "i": i,
                    "from": i,
                    "to": i + 1,
                    "t": t,
                    "x": float(x),
                    "y": float(y3),
                    "z": float(z3),
                    **audio_stats(parts[-1]),
                }
            )
        audio_path = AUDIO / f"{name}.wav"
        sf.write(audio_path, join(parts), SR)
        payload["variants"].append(
            {"id": name, "label": label, "color": color, "audio": f"audio/{audio_path.name}", "route": route}
        )
    return payload


def html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nightjar Latent Route</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f6f8;color:#15171a}}
main{{height:100vh;display:grid;grid-template-columns:minmax(0,1fr) 360px}}#plot{{height:100vh}}
aside{{padding:22px;background:#fff;border-left:1px solid #d9dde3;display:flex;flex-direction:column;gap:16px}}
h1{{font-size:22px;line-height:1.15;margin:0 0 8px}}p{{margin:0;color:#626975;line-height:1.45}}button{{font:inherit}}
.variant{{width:100%;padding:10px 12px;margin:0;border:1px solid #cfd5dd;border-radius:8px;background:#fbfcfd;text-align:left;cursor:pointer}}
.variant.active{{border-color:var(--c);box-shadow:inset 4px 0 0 var(--c);background:#fff}}
.variant-row{{display:grid;grid-template-columns:minmax(0,1fr) 38px;gap:8px;align-items:center;margin:4px 0}}.compare{{height:38px;border:1px solid #cfd5dd;border-radius:8px;background:#f8fafc;display:grid;place-items:center}}
.compare input{{width:16px;height:16px;accent-color:#15171a}}
.label{{font-size:12px;font-weight:600;color:#626975;margin-bottom:-8px}}.tools{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.tools.two{{grid-template-columns:1fr 1fr}}.tools.one{{grid-template-columns:1fr}}
.tools button{{height:34px;border:1px solid #cfd5dd;border-radius:8px;background:#f8fafc;cursor:pointer}}
.tools button.active{{background:#15171a;color:#fff;border-color:#15171a}}audio{{width:100%}}
@media(max-width:900px){{main{{display:block}}#plot{{height:68vh}}aside{{border-left:0;border-top:1px solid #d9dde3}}}}
</style></head><body><main><div id="plot"></div><aside>
<div><h1>Nightjar latent route</h1><p>Encoded Nightjar anchors and generated near-anchor routes.</p></div>
<div id="buttons"></div><audio id="audio" controls preload="metadata"></audio>
<div class="label">View</div><div class="tools"><button class="view active" data-view="3d">3D</button><button class="view" data-view="xy">PC1/2</button><button class="view" data-view="xz">PC1/3</button><button class="view" data-view="yz">PC2/3</button></div>
<div class="label">Scale</div><div class="tools two"><button class="scale active" data-scale="norm">Normalized</button><button class="scale" data-scale="raw">Raw</button></div>
<div class="label">Range</div><div class="tools two"><button class="range active" data-range="focus">Focus</button><button class="range" data-range="all">All</button></div>
<div class="label">Layers</div><div class="tools one"><button id="anchorPath" class="active">Anchor path</button></div>
<div class="tools"><button id="orbit" class="active">Orbit</button><button id="pan">Pan</button><button id="reset">Reset</button><button id="follow">Follow</button></div>
</aside></main><script>
const DATA={data}, plot=document.getElementById('plot'), audio=document.getElementById('audio'), buttons=document.getElementById('buttons');
let active=DATA.variants[0], view='3d', scaleMode='norm', rangeMode='focus', dragMode='orbit', currentTrace=DATA.variants.length+2, followPlayback=false, showAnchorPath=true, savedCamera=null, cameraUpdateDepth=0;
const compare=new Set();
const axes={{x:'PC1',y:'PC2',z:'PC3'}}, views={{'3d':['x','y','z'],xy:['x','y'],xz:['x','z'],yz:['y','z']}};
const all=[...DATA.anchors,...DATA.variants.flatMap(v=>v.route)], pad=.08;
function values(key){{return all.map(p=>p[key]).sort((a,b)=>a-b);}}
function quantile(a,q){{let i=(a.length-1)*q,lo=Math.floor(i),hi=Math.ceil(i),t=i-lo;return a[lo]*(1-t)+a[hi]*t;}}
const stats=Object.fromEntries(['x','y','z'].map(k=>{{let a=values(k),s=(quantile(a,.75)-quantile(a,.25))/1.35||1;return [k,{{center:quantile(a,.5),scale:s}}];}}));
function coord(p,key){{if(scaleMode==='raw')return p[key];let s=stats[key];return Math.asinh((p[key]-s.center)/s.scale);}}
function routeVisible(v){{return v.id===active.id||compare.has(v.id);}}
function routeSet(){{return DATA.variants.filter(routeVisible).flatMap(v=>v.route);}}
function rangeSet(){{return [...DATA.anchors,...routeSet()];}}
function sortedValues(points,key){{return points.map(p=>coord(p,key)).sort((a,b)=>a-b);}}
function focusValues(key){{let anchors=sortedValues(DATA.anchors,key),lo=quantile(anchors,.12),hi=quantile(anchors,.88),central=DATA.anchors.filter(p=>{{let v=coord(p,key);return v>=lo&&v<=hi;}});return sortedValues([...central,...routeSet()],key);}}
function makeRange(key,points,loQ=0,hiQ=1){{let a=points,lo=quantile(a,loQ),hi=quantile(a,hiQ),d=(hi-lo)||1;return [lo-d*pad,hi+d*pad];}}
function ranges(){{return {{all:{{x:makeRange('x',sortedValues(rangeSet(),'x')),y:makeRange('y',sortedValues(rangeSet(),'y')),z:makeRange('z',sortedValues(rangeSet(),'z'))}},focus:{{x:makeRange('x',focusValues('x')),y:makeRange('y',focusValues('y')),z:makeRange('z',focusValues('z'))}}}};}}
const defaultCamera={{eye:{{x:1.55,y:1.7,z:1.15}},up:{{x:0,y:0,z:1}}}};
const config={{responsive:true,displayModeBar:true,scrollZoom:true,modeBarButtonsToAdd:['orbitRotation','pan3d','resetCameraDefault3d'],displaylogo:false}};
function cloneCamera(c){{return c?JSON.parse(JSON.stringify(c)):null;}}
function currentCamera(){{return view==='3d'&&plot._fullLayout&&plot._fullLayout.scene?cloneCamera(plot._fullLayout.scene.camera):null;}}
function rememberCamera(){{let c=currentCamera();if(c)savedCamera=c;}}
function guardCamera(work){{cameraUpdateDepth++;return Promise.resolve(work()).finally(()=>{{cameraUpdateDepth=Math.max(0,cameraUpdateDepth-1);}});}}
function keepCamera(work){{let c=savedCamera||currentCamera();return guardCamera(()=>Promise.resolve(work()).then(()=>{{if(view==='3d'&&(savedCamera||c))return Plotly.relayout(plot,{{'scene.camera':savedCamera||c}});}}));}}
function relayout(update){{return guardCamera(()=>Plotly.relayout(plot,update));}}
function updateMarker(restyle,p){{let patch=followPlayback?followLayout(p):{{}},c=savedCamera||currentCamera();if(view==='3d'&&c)patch['scene.camera']=c;return guardCamera(()=>Plotly.update(plot,restyle,patch,[markerIndex()]));}}
function dims(){{return views[view];}}
function axisTitle(key){{return scaleMode==='norm'?`n${{axes[key]}}`:axes[key];}}
function inFocus(p){{let r=ranges().focus;return dims().every(k=>coord(p,k)>=r[k][0]&&coord(p,k)<=r[k][1]);}}
function visible(points){{return points;}}
function textForRoute(p){{return `Segment ${{p.i}}<br>source ${{p.from}}-${{p.to}}<br>latent t=${{p.t.toFixed(2)}}<br>rms=${{p.rms.toFixed(3)}}<br>centroid=${{Math.round(p.centroid)}} Hz`;}}
function baseTrace(points,name,color,dash,shown=true,kind='route'){{let d=dims(),is3=view==='3d',anchor=kind==='anchor',anchorLine=kind==='anchorPath',pts=visible(points),trace={{x:pts.map(p=>coord(p,d[0])),y:pts.map(p=>coord(p,d[1])),mode:anchor?'markers':'lines+markers',type:is3?'scatter3d':'scatter',name,visible:shown,line:{{color,width:anchor?0:anchorLine?2.5:3.5,dash:dash||'solid'}},marker:{{color,size:anchor?4:anchorLine?4:is3?5:7,opacity:anchor ? .72 : .86}},text:pts.map(p=>p.time!==undefined?`Anchor ${{p.i}}<br>${{p.time.toFixed(1)}} s`:textForRoute(p)),hovertemplate:'%{{text}}<extra></extra>',connectgaps:true}};if(is3)trace.z=pts.map(p=>coord(p,d[2]));return trace;}}
function currentPointTrace(p){{let d=dims(),is3=view==='3d',trace={{x:[coord(p,d[0])],y:[coord(p,d[1])],mode:'markers',type:is3?'scatter3d':'scatter',name:'Playback position',marker:{{color:'#fff',size:is3?10:14,line:{{color:active.color,width:5}}}},hoverinfo:'skip'}};if(is3)trace.z=[coord(p,d[2])];return trace;}}
function axis2d(title,range){{return {{title,range,showgrid:true,gridcolor:'#d6dbe2',gridwidth:1,zeroline:true,zerolinecolor:'#8f98a3',zerolinewidth:2,showline:true,linecolor:'#a9b0ba',linewidth:1,mirror:true,ticks:'outside',tickcolor:'#8f98a3',ticklen:6}};}}
function layout(){{let d=dims(),r=ranges()[rangeMode],is3=view==='3d',base={{dragmode:is3?undefined:'pan',margin:{{l:44,r:12,b:42,t:12}},paper_bgcolor:'#f5f6f8',plot_bgcolor:'#f5f6f8',uirevision:'nightjar-route',legend:{{x:.02,y:.98,bgcolor:'rgba(255,255,255,.72)'}}}};if(is3){{base.margin={{l:0,r:0,b:0,t:0}};base.scene={{uirevision:'nightjar-camera',dragmode:dragMode,xaxis:{{title:axisTitle(d[0]),range:r[d[0]],showbackground:false,gridcolor:'#d4d8df'}},yaxis:{{title:axisTitle(d[1]),range:r[d[1]],showbackground:false,gridcolor:'#d4d8df'}},zaxis:{{title:axisTitle(d[2]),range:r[d[2]],showbackground:false,gridcolor:'#d4d8df'}},camera:savedCamera||defaultCamera,aspectmode:'cube'}};}}else {{base.xaxis=axis2d(axisTitle(d[0]),r[d[0]]);base.yaxis={{...axis2d(axisTitle(d[1]),r[d[1]]),scaleanchor:'x',scaleratio:1}};}}return base;}}
function traces(){{let p=point()||active.route[0];currentTrace=DATA.variants.length+2;return [baseTrace(DATA.anchors,'Nightjar anchors','#171717','solid',true,'anchor'),baseTrace(DATA.anchors,'Nightjar anchor path','#171717','solid',showAnchorPath,'anchorPath'),...DATA.variants.map(v=>baseTrace(v.route,v.label,v.color,'solid',routeVisible(v))),currentPointTrace(p)];}}
function cameraEvent(e){{return e&&Object.keys(e).some(k=>k==='scene.camera'||k.startsWith('scene.camera'));}}
function saveCameraEvent(e){{if(!cameraUpdateDepth&&cameraEvent(e))rememberCamera();}}
Plotly.newPlot(plot,traces(),layout(),config).then(()=>{{plot.on('plotly_relayout',saveCameraEvent);plot.on('plotly_relayouting',saveCameraEvent);}});
function syncVariantControls(){{document.querySelectorAll('.variant').forEach(b=>b.classList.toggle('active',b.dataset.id===active.id));document.querySelectorAll('.compare input').forEach(c=>{{c.checked=c.dataset.id===active.id||compare.has(c.dataset.id);c.disabled=c.dataset.id===active.id;}});}}
function setVariant(id){{active=DATA.variants.find(v=>v.id===id)||DATA.variants[0];audio.src=active.audio;syncVariantControls();redraw();}}
DATA.variants.forEach(v=>{{let row=document.createElement('div'),b=document.createElement('button'),label=document.createElement('label'),c=document.createElement('input');row.className='variant-row';b.className='variant';b.dataset.id=v.id;b.style.setProperty('--c',v.color);b.textContent=v.label;b.onclick=()=>setVariant(v.id);c.type='checkbox';c.dataset.id=v.id;c.onchange=()=>{{c.checked?compare.add(v.id):compare.delete(v.id);syncVariantControls();redraw();}};label.className='compare';label.title='Compare';label.appendChild(c);row.append(b,label);buttons.appendChild(row);}});
function applyDragMode(){{syncDragButtons();return view==='3d'?relayout({{'scene.dragmode':dragMode}}):relayout({{dragmode:'pan'}});}}
function redraw(){{keepCamera(()=>Plotly.react(plot,traces(),layout(),config)).then(()=>applyDragMode()).then(()=>draw());}}
function syncDragButtons(){{orbit.classList.toggle('active',view==='3d'&&dragMode==='orbit');pan.classList.toggle('active',view!=='3d'||dragMode==='pan');}}
function setView(next){{view=next;document.querySelectorAll('.view').forEach(b=>b.classList.toggle('active',b.dataset.view===view));syncDragButtons();redraw();}}
function setScale(next){{scaleMode=next;document.querySelectorAll('.scale').forEach(b=>b.classList.toggle('active',b.dataset.scale===scaleMode));redraw();}}
function setRange(next){{rangeMode=next;document.querySelectorAll('.range').forEach(b=>b.classList.toggle('active',b.dataset.range===rangeMode));redraw();}}
function setDrag(mode){{dragMode=mode;applyDragMode();}}
function resetView(){{savedCamera=null;followPlayback=false;followButton();redraw();}}
function followButton(){{document.getElementById('follow').classList.toggle('active',followPlayback);}}
function point(){{let r=active.route,n=r.length;if(!n)return null;let d=Number.isFinite(audio.duration)&&audio.duration>0?audio.duration:(n*DATA.meta.segmentSec),u=Math.max(0,Math.min(.9999,audio.currentTime/d))*(n-1),i=Math.floor(u),a=u-i,p=r[i],q=r[Math.min(i+1,n-1)];return {{i:p.i,from:p.from,to:p.to,t:p.t,centroid:p.centroid,x:p.x+(q.x-p.x)*a,y:p.y+(q.y-p.y)*a,z:p.z+(q.z-p.z)*a}};}}
function followLayout(p){{let d=dims(),r=ranges()[rangeMode],out={{}};d.forEach((key,i)=>{{let c=coord(p,key),width=(r[key][1]-r[key][0])*.42,outKey=view==='3d'?`scene.${{['x','y','z'][i]}}axis.range`:(i===0?'xaxis.range':'yaxis.range');out[outKey]=[c-width,c+width];}});return out;}}
function markerIndex(){{let i=(plot.data||[]).findIndex(t=>t.name==='Playback position');return i>=0?i:currentTrace;}}
function draw(){{let p=point();if(!p)return;let d=dims(),is3=view==='3d',restyle={{x:[[coord(p,d[0])]],y:[[coord(p,d[1])]],'marker.line.color':active.color}};if(is3)restyle.z=[[coord(p,d[2])]];updateMarker(restyle,p);}}
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>setView(b.dataset.view));
document.querySelectorAll('.scale').forEach(b=>b.onclick=()=>setScale(b.dataset.scale));
document.querySelectorAll('.range').forEach(b=>b.onclick=()=>setRange(b.dataset.range));
document.getElementById('anchorPath').onclick=()=>{{showAnchorPath=!showAnchorPath;document.getElementById('anchorPath').classList.toggle('active',showAnchorPath);redraw();}};
document.getElementById('orbit').onclick=()=>setDrag('orbit');document.getElementById('pan').onclick=()=>setDrag('pan');document.getElementById('reset').onclick=resetView;document.getElementById('follow').onclick=()=>{{followPlayback=!followPlayback;followButton();draw();}};
audio.ontimeupdate=draw;audio.onplay=()=>requestAnimationFrame(function loop(){{draw();if(!audio.paused&&!audio.ended)requestAnimationFrame(loop);}});
syncVariantControls();
setVariant(active.id);
</script></body></html>"""


def main() -> dict:
    payload = render()
    HTML.write_text(html(payload), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = main()
    print(f"Wrote {HTML}")
    print(f"Variants: {len(result['variants'])}")
