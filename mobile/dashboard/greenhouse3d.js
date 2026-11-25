/**
 * ADA2025 甜菜温室 · 建筑与植物形态学级 WebGL Three.js 写实 3D 数字孪生引擎
 * 核心特性：
 * 1. 几何微细节：3段圆角倒角(Bevel) + 高密度细分曲面(Subdivision) + 三维分形布朗噪声(3D FBM)顶点位移(Displacement)
 * 2. PBR 物理材质：全通道 Albedo/Roughness/Normal/AO 贴图 + 边缘磨损(Edge Wear)与表面水垢微瑕(Surface Imperfections)
 * 3. 真实光照与阴影：程序化 HDRI 全局照明(PMREM GI) + 冷暖对比三点布光(软阴影主光+冷色补光+双轮廓逆光) + 丁达尔体积雾
 * 4. 渲染与相机质感：ACES Filmic 色调映射 + 50mm 标准镜头视角(35° FOV) + WebGL 实时散景景深(Depth of Field)后处理管线
 */
(function (global) {
  'use strict';

  // 确定性伪随机生成器（保证每株甜菜形态自然差异且跨刷新一致）
  function mulberry32(seed) {
    return function () {
      var t = (seed += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // 三维连续值噪声与 3 阶分形布朗运动 (3D FBM Noise)，用于真实物理表面几何顶点微位移
  function hash3(ix, iy, iz, seed) {
    var n = ix * 374761393 + iy * 668265263 + iz * 2147483647 + (seed || 0) * 1274126177;
    n = (n ^ (n >> 13)) * 1274126177;
    return ((n ^ (n >> 16)) >>> 0) / 4294967296;
  }
  function valueNoise3D(x, y, z, seed) {
    var ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
    var fx = x - ix, fy = y - iy, fz = z - iz;
    var ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy), uz = fz * fz * (3 - 2 * fz);
    function lerp(a, b, t) { return a + (b - a) * t; }
    var c000 = hash3(ix, iy, iz, seed), c100 = hash3(ix + 1, iy, iz, seed);
    var c010 = hash3(ix, iy + 1, iz, seed), c110 = hash3(ix + 1, iy + 1, iz, seed);
    var c001 = hash3(ix, iy, iz + 1, seed), c101 = hash3(ix + 1, iy, iz + 1, seed);
    var c011 = hash3(ix, iy + 1, iz + 1, seed), c111 = hash3(ix + 1, iy + 1, iz + 1, seed);
    var x00 = lerp(c000, c100, ux), x10 = lerp(c010, c110, ux);
    var x01 = lerp(c001, c101, ux), x11 = lerp(c011, c111, ux);
    var y0 = lerp(x00, x10, uy), y1 = lerp(x01, x11, uy);
    return lerp(y0, y1, uz) * 2 - 1;
  }
  function fbm3D(x, y, z, seed) {
    return (
      0.58 * valueNoise3D(x, y, z, seed) +
      0.28 * valueNoise3D(x * 2.13, y * 2.13, z * 2.13, seed + 17) +
      0.14 * valueNoise3D(x * 4.41, y * 4.41, z * 4.41, seed + 37)
    );
  }

  // 为任意 BufferGeometry 补齐 uv2 通道（激活 aoMap），并施加 3D FBM 微小几何起伏位移
  function prepareAndDisplaceGeometry(geo, ampX, ampY, ampZ, freq, seed) {
    if (geo.attributes.uv && !geo.attributes.uv2) {
      geo.setAttribute('uv2', geo.attributes.uv.clone());
    }
    if (ampX || ampY || ampZ) {
      var pos = geo.attributes.position;
      var f = freq || 4.0;
      for (var i = 0; i < pos.count; i++) {
        var vx = pos.getX(i), vy = pos.getY(i), vz = pos.getZ(i);
        var nx = fbm3D(vx * f, vy * f, vz * f, seed || 101);
        var ny = fbm3D(vx * f + 7.1, vy * f + 3.4, vz * f, (seed || 101) + 23);
        var nz = fbm3D(vx * f, vy * f + 9.2, vz * f + 5.7, (seed || 101) + 59);
        pos.setXYZ(i, vx + nx * (ampX || 0), vy + ny * (ampY || 0), vz + nz * (ampZ || 0));
      }
      pos.needsUpdate = true;
      geo.computeVertexNormals();
    }
    return geo;
  }

  // 由灰度高度画布生成切线空间 RGB 法线贴图 (Sobel Normal Map)
  function heightCanvasToNormalTex(hCanvas, strength, repeatX, repeatY) {
    var sz = hCanvas.width;
    var src = hCanvas.getContext('2d').getImageData(0, 0, sz, sz).data;
    var cNorm = document.createElement('canvas'); cNorm.width = sz; cNorm.height = sz;
    var gn = cNorm.getContext('2d');
    var nImg = gn.createImageData(sz, sz);
    var str = strength || 2.4;
    for (var y = 0; y < sz; y++) {
      for (var x = 0; x < sz; x++) {
        var idx = (y * sz + x) * 4;
        var xm = (x - 1 + sz) % sz, xp = (x + 1) % sz;
        var ym = (y - 1 + sz) % sz, yp = (y + 1) % sz;
        var hL = src[(y * sz + xm) * 4] / 255;
        var hR = src[(y * sz + xp) * 4] / 255;
        var hU = src[(ym * sz + x) * 4] / 255;
        var hD = src[(yp * sz + x) * 4] / 255;
        var dx = (hL - hR) * str;
        var dy = (hU - hD) * str;
        var dz = 1.0;
        var inv = 1.0 / Math.hypot(dx, dy, dz);
        nImg.data[idx] = Math.floor((dx * inv * 0.5 + 0.5) * 255);
        nImg.data[idx + 1] = Math.floor((dy * inv * 0.5 + 0.5) * 255);
        nImg.data[idx + 2] = Math.floor((dz * inv * 0.5 + 0.5) * 255);
        nImg.data[idx + 3] = 255;
      }
    }
    gn.putImageData(nImg, 0, 0);
    var nTex = new THREE.CanvasTexture(cNorm);
    nTex.anisotropy = 8;
    if (repeatX || repeatY) {
      nTex.wrapS = nTex.wrapT = THREE.RepeatWrapping;
      nTex.repeat.set(repeatX || 1, repeatY || 1);
    }
    return nTex;
  }

  // 生成带边缘磨损 (Edge Wear)、水垢微瑕 (Surface Imperfections)、法线微凹凸与 AO 的全套工业级 PBR 贴图组
  function makeWeatheredPBRSet(opts) {
    var sz = opts.size || 256;
    var baseRGB = opts.baseRGB || [185, 196, 204];
    var wearRGB = opts.wearRGB || [232, 238, 242];
    var dirtRGB = opts.dirtRGB || [92, 86, 78];
    var seed = opts.seed || 1001;
    var repX = opts.repeatX || 2, repY = opts.repeatY || 2;
    var rng = mulberry32(seed);

    var cAlb = document.createElement('canvas'); cAlb.width = sz; cAlb.height = sz;
    var ga = cAlb.getContext('2d');
    var cRough = document.createElement('canvas'); cRough.width = sz; cRough.height = sz;
    var gr = cRough.getContext('2d');
    var cHeight = document.createElement('canvas'); cHeight.width = sz; cHeight.height = sz;
    var gh = cHeight.getContext('2d');
    var cAo = document.createElement('canvas'); cAo.width = sz; cAo.height = sz;
    var gao = cAo.getContext('2d');

    ga.fillStyle = 'rgb(' + baseRGB[0] + ',' + baseRGB[1] + ',' + baseRGB[2] + ')';
    ga.fillRect(0, 0, sz, sz);
    var baseRoughByte = Math.floor((opts.roughness || 0.5) * 255);
    gr.fillStyle = 'rgb(' + baseRoughByte + ',' + baseRoughByte + ',' + baseRoughByte + ')';
    gr.fillRect(0, 0, sz, sz);
    gh.fillStyle = '#808080'; gh.fillRect(0, 0, sz, sz);
    gao.fillStyle = '#f2f2f2'; gao.fillRect(0, 0, sz, sz);

    // 1. 低频斑驳肌理与温室水雾钙质水垢斑块 (Surface Imperfections)
    for (var i = 0; i < 200; i++) {
      var px = rng() * sz, py = rng() * sz, rad = 8 + rng() * 28;
      var isDirt = rng() > 0.42;
      var col = isDirt ? dirtRGB : wearRGB;
      var alpha = (0.04 + rng() * 0.08).toFixed(3);
      ga.fillStyle = 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + alpha + ')';
      ga.beginPath(); ga.arc(px, py, rad, 0, Math.PI * 2); ga.fill();

      var rShift = isDirt ? Math.min(252, baseRoughByte + 38) : Math.max(45, baseRoughByte - 34);
      gr.fillStyle = 'rgba(' + rShift + ',' + rShift + ',' + rShift + ',0.24)';
      gr.beginPath(); gr.arc(px, py, rad, 0, Math.PI * 2); gr.fill();

      var hShift = isDirt ? 110 : 148;
      gh.fillStyle = 'rgba(' + hShift + ',' + hShift + ',' + hShift + ',0.22)';
      gh.beginPath(); gh.arc(px, py, rad * 0.8, 0, Math.PI * 2); gh.fill();
    }

    // 2. 高频微晶颗粒/金属拉丝与微细划痕 (Micro-scratches & Brushed Grain)
    for (var j = 0; j < 1800; j++) {
      var gx = rng() * sz, gy = rng() * sz;
      var delta = Math.floor((rng() - 0.5) * 26);
      var r = Math.min(255, Math.max(0, baseRGB[0] + delta));
      var g = Math.min(255, Math.max(0, baseRGB[1] + delta));
      var b = Math.min(255, Math.max(0, baseRGB[2] + delta));
      ga.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.28)';
      ga.fillRect(gx, gy, 1.5 + rng() * 3, 1.5 + rng() * 2);

      var hVal = Math.min(255, Math.max(0, 128 + delta * 1.6));
      gh.fillStyle = 'rgb(' + hVal + ',' + hVal + ',' + hVal + ')';
      gh.fillRect(gx, gy, 1.8, 1.8);
    }

    for (var s = 0; s < 48; s++) {
      var sx = rng() * sz, sy = rng() * sz, ang = rng() * Math.PI, len = 10 + rng() * 26;
      var ex = sx + Math.cos(ang) * len, ey = sy + Math.sin(ang) * len;
      ga.strokeStyle = 'rgba(' + wearRGB[0] + ',' + wearRGB[1] + ',' + wearRGB[2] + ',0.22)';
      ga.lineWidth = 0.9;
      ga.beginPath(); ga.moveTo(sx, sy); ga.lineTo(ex, ey); ga.stroke();
      gh.strokeStyle = 'rgba(90,90,90,0.45)';
      gh.lineWidth = 1.1;
      gh.beginPath(); gh.moveTo(sx, sy); gh.lineTo(ex, ey); gh.stroke();
      gr.strokeStyle = 'rgba(85,85,85,0.35)';
      gr.lineWidth = 1.0;
      gr.beginPath(); gr.moveTo(sx, sy); gr.lineTo(ex, ey); gr.stroke();
    }

    // 3. 边缘磨损 (Edge Wear) 与接缝环境光遮蔽 (Crevice AO)
    var border = Math.max(6, Math.floor(sz * 0.055));
    gao.strokeStyle = 'rgba(65,65,65,0.55)';
    gao.lineWidth = border * 1.4;
    gao.strokeRect(0, 0, sz, sz);
    for (var w = 0; w < 80; w++) {
      var onVert = rng() > 0.5;
      var wx = onVert ? (rng() > 0.5 ? rng() * border : sz - rng() * border) : rng() * sz;
      var wy = onVert ? rng() * sz : (rng() > 0.5 ? rng() * border : sz - rng() * border);
      ga.fillStyle = 'rgba(' + wearRGB[0] + ',' + wearRGB[1] + ',' + wearRGB[2] + ',0.35)';
      ga.fillRect(wx - 2, wy - 2, 4 + rng() * 6, 4 + rng() * 6);
      gr.fillStyle = 'rgba(75,75,75,0.42)';
      gr.fillRect(wx - 2, wy - 2, 4 + rng() * 6, 4 + rng() * 6);
    }

    function toTex(cv, isSRGB) {
      var t = new THREE.CanvasTexture(cv);
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(repX, repY);
      t.anisotropy = 8;
      if (isSRGB) t.encoding = THREE.sRGBEncoding;
      return t;
    }

    return {
      map: toTex(cAlb, true),
      roughnessMap: toTex(cRough, false),
      normalMap: heightCanvasToNormalTex(cHeight, opts.normalStrength || 2.2, repX, repY),
      aoMap: toTex(cAo, false)
    };
  }

  /* ==================== 1. 程序化高清 PBR 纹理工厂 ==================== */

  // 建筑级钢化玻璃斜向天光反射与边框胶条纹理（彻底解决“温室有架无皮”问题）
  function makeGlassReflectionTex() {
    var sz = 512;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    g.fillStyle = 'rgba(212,236,248,0.34)';
    g.fillRect(0, 0, sz, sz);

    // 斜向菲涅尔天空高光反射光带
    var sheen1 = g.createLinearGradient(0, 0, sz, sz);
    sheen1.addColorStop(0.00, 'rgba(255,255,255,0.05)');
    sheen1.addColorStop(0.24, 'rgba(255,255,255,0.42)');
    sheen1.addColorStop(0.36, 'rgba(220,242,255,0.08)');
    sheen1.addColorStop(0.62, 'rgba(255,255,255,0.32)');
    sheen1.addColorStop(0.75, 'rgba(210,235,250,0.06)');
    sheen1.addColorStop(1.00, 'rgba(255,255,255,0.22)');
    g.fillStyle = sheen1;
    g.fillRect(0, 0, sz, sz);

    // 分格玻璃铝合金压条与黑色三元乙丙密封胶条边缘
    var panes = 4;
    var step = sz / panes;
    for (var i = 0; i <= panes; i++) {
      var px = Math.min(sz - 2, Math.max(2, i * step));
      g.strokeStyle = 'rgba(54,68,78,0.52)';
      g.lineWidth = 3.5;
      g.beginPath(); g.moveTo(px, 0); g.lineTo(px, sz); g.stroke();
      g.strokeStyle = 'rgba(235,244,250,0.68)';
      g.lineWidth = 1.4;
      g.beginPath(); g.moveTo(px + 1.5, 0); g.lineTo(px + 1.5, sz); g.stroke();
    }
    g.strokeStyle = 'rgba(54,68,78,0.55)';
    g.lineWidth = 5;
    g.strokeRect(2, 2, sz - 4, sz - 4);

    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3, 1);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    return tex;
  }

  function makeBeetLeafTextures() {
    var sz = 512;
    var cMap = document.createElement('canvas'); cMap.width = sz; cMap.height = sz;
    var g = cMap.getContext('2d');

    // 叶面底色：白色基准（供 leafMat.color 染色）叠加真实叶绿素细胞斑驳与叶柄微粉红过渡
    var baseGrad = g.createLinearGradient(0, sz, 0, 0);
    baseGrad.addColorStop(0.00, '#e8c4c0'); // 甜菜基部叶柄特有的微紫红/嫩粉色过渡
    baseGrad.addColorStop(0.24, '#e5f2d8'); // 叶柄至叶基浅嫩绿
    baseGrad.addColorStop(0.45, '#f2f8ec');
    baseGrad.addColorStop(1.00, '#ffffff');
    g.fillStyle = baseGrad;
    g.fillRect(0, 0, sz, sz);

    // 细胞级叶肉明暗肌理
    var rng = mulberry32(202611);
    for (var i = 0; i < 6500; i++) {
      var px = rng() * sz, py = rng() * sz;
      var lum = 225 + Math.floor(rng() * 30);
      g.fillStyle = 'rgba(' + lum + ',' + Math.min(255, lum + 5) + ',' + (lum - 8) + ',0.18)';
      g.fillRect(px, py, 2 + rng() * 3, 2 + rng() * 3);
    }

    // 凹凸贴图画布（叶脉隆起）
    var cBump = document.createElement('canvas'); cBump.width = sz; cBump.height = sz;
    var gb = cBump.getContext('2d');
    gb.fillStyle = '#707070'; gb.fillRect(0, 0, sz, sz);

    // 主脉（粗壮微黄白，向上逐渐变细）
    var midGrad = g.createLinearGradient(0, sz, 0, 32);
    midGrad.addColorStop(0, '#f4d8d2');
    midGrad.addColorStop(0.3, '#f6ffe8');
    midGrad.addColorStop(1, '#e6f5d5');
    for (var step = 0; step < 36; step++) {
      var t0 = step / 36, t1 = (step + 1) / 36;
      var y0 = sz - t0 * (sz - 34), y1 = sz - t1 * (sz - 34);
      var w = (1 - t0) * 14 + 2.2;
      g.strokeStyle = midGrad; g.lineWidth = w; g.lineCap = 'round';
      g.beginPath(); g.moveTo(sz / 2, y0); g.lineTo(sz / 2, y1); g.stroke();
      gb.strokeStyle = '#ececec'; gb.lineWidth = w; gb.lineCap = 'round';
      gb.beginPath(); gb.moveTo(sz / 2, y0); gb.lineTo(sz / 2, y1); gb.stroke();
    }

    // 二级羽状侧脉 + 三级细脉
    for (var vi = 0; vi < 11; vi++) {
      var vy = sz * 0.78 - vi * (sz * 0.064);
      var span = Math.sin((vi + 1.5) / 13 * Math.PI) * (sz * 0.44);
      var rise = 38 + vi * 3;
      [-1, 1].forEach(function (dir) {
        var ex = sz / 2 + dir * span;
        var ey = Math.max(24, vy - rise);
        var cx = sz / 2 + dir * span * 0.55;
        var cy = vy - rise * 0.25;
        var lw = Math.max(1.5, 4.5 - vi * 0.25);

        g.strokeStyle = 'rgba(244,255,228,0.78)'; g.lineWidth = lw;
        g.beginPath(); g.moveTo(sz / 2, vy); g.quadraticCurveTo(cx, cy, ex, ey); g.stroke();
        gb.strokeStyle = 'rgba(225,225,225,0.85)'; gb.lineWidth = lw;
        gb.beginPath(); gb.moveTo(sz / 2, vy); gb.quadraticCurveTo(cx, cy, ex, ey); gb.stroke();

        // 分叉细脉
        g.strokeStyle = 'rgba(235,248,220,0.45)'; g.lineWidth = lw * 0.55;
        g.beginPath(); g.moveTo(cx, cy); g.lineTo(ex - dir * 18, ey - 22); g.stroke();
        gb.strokeStyle = 'rgba(195,195,195,0.5)'; gb.lineWidth = lw * 0.55;
        gb.beginPath(); gb.moveTo(cx, cy); gb.lineTo(ex - dir * 18, ey - 22); gb.stroke();
      });
    }

    // 真实植物生理向心失绿黄化纹理 (mapChlorosis)：中脉与叶基保留叶绿素绿意，叶尖与叶缘渐变为金黄与焦褐斑驳（彻底拒绝一刀切刷黄漆！）
    var cChl = document.createElement('canvas'); cChl.width = sz; cChl.height = sz;
    var gc = cChl.getContext('2d');
    gc.drawImage(cMap, 0, 0);
    // 中脉及叶基保持鲜活叶绿素底色，两侧叶缘与叶尖叠加向心失绿渐变
    var radChl = gc.createRadialGradient(sz * 0.5, sz * 0.68, sz * 0.08, sz * 0.5, sz * 0.46, sz * 0.48);
    radChl.addColorStop(0.00, 'rgba(62,128,44,0.38)');   // 叶基与中脉区保留生理绿意
    radChl.addColorStop(0.48, 'rgba(138,162,42,0.32)');  // 中间过渡带黄绿
    radChl.addColorStop(0.78, 'rgba(224,182,44,0.58)');  // 叶缘失绿金黄
    radChl.addColorStop(1.00, 'rgba(176,104,32,0.74)');  // 叶尖与最外缘干枯焦褐
    gc.fillStyle = radChl;
    gc.fillRect(0, 0, sz, sz);
    // 叠加不规则叶脉间失绿斑块
    for (var cp = 0; cp < 75; cp++) {
      var cpx = rng() * sz, cpy = rng() * sz * 0.78;
      var distMid = Math.abs(cpx - sz * 0.5) / (sz * 0.5);
      if (distMid > 0.22 || cpy < sz * 0.28) {
        var cr = 8 + rng() * 22;
        gc.fillStyle = rng() > 0.3 ? 'rgba(232,188,48,0.22)' : 'rgba(162,96,30,0.20)';
        gc.beginPath(); gc.arc(cpx, cpy, cr, 0, Math.PI * 2); gc.fill();
      }
    }

    var mapTex = new THREE.CanvasTexture(cMap);
    mapTex.encoding = THREE.sRGBEncoding;
    mapTex.anisotropy = 8;
    var mapChlTex = new THREE.CanvasTexture(cChl);
    mapChlTex.encoding = THREE.sRGBEncoding;
    mapChlTex.anisotropy = 8;
    var bumpTex = new THREE.CanvasTexture(cBump);
    bumpTex.anisotropy = 8;

    // 由高度图 Sobel 求导生成真实切线空间 RGB 法线贴图 (NormalMap) + 粗糙度变化贴图 (RoughnessMap) + 根部遮蔽 (AOMap)
    var srcData = gb.getImageData(0, 0, sz, sz).data;
    var cNorm = document.createElement('canvas'); cNorm.width = sz; cNorm.height = sz;
    var gn = cNorm.getContext('2d');
    var nImg = gn.createImageData(sz, sz);
    var cRough = document.createElement('canvas'); cRough.width = sz; cRough.height = sz;
    var gr = cRough.getContext('2d');
    var rImg = gr.createImageData(sz, sz);
    var cAo = document.createElement('canvas'); cAo.width = sz; cAo.height = sz;
    var gao = cAo.getContext('2d');
    var aoGrad = gao.createLinearGradient(0, sz, 0, 0);
    aoGrad.addColorStop(0.0, '#484848'); // 叶柄根部深处强环境光遮蔽 (AO)
    aoGrad.addColorStop(0.28, '#b0b0b0');
    aoGrad.addColorStop(0.65, '#f0f0f0');
    aoGrad.addColorStop(1.0, '#ffffff');
    gao.fillStyle = aoGrad; gao.fillRect(0, 0, sz, sz);

    var rngR = mulberry32(7788);
    for (var y = 0; y < sz; y++) {
      for (var x = 0; x < sz; x++) {
        var idx = (y * sz + x) * 4;
        var xm = Math.max(0, x - 1), xp = Math.min(sz - 1, x + 1);
        var ym = Math.max(0, y - 1), yp = Math.min(sz - 1, y + 1);
        var hL = srcData[(y * sz + xm) * 4] / 255;
        var hR = srcData[(y * sz + xp) * 4] / 255;
        var hU = srcData[(ym * sz + x) * 4] / 255;
        var hD = srcData[(yp * sz + x) * 4] / 255;
        // 叠加低频有机噪点打破平整感
        var lfNoise = Math.sin(x * 0.045) * Math.cos(y * 0.045) * 0.12;
        var dx = (hL - hR) * 2.8 + lfNoise;
        var dy = (hU - hD) * 2.8 + lfNoise;
        var dz = 1.0;
        var inv = 1.0 / Math.hypot(dx, dy, dz);
        nImg.data[idx] = Math.floor((dx * inv * 0.5 + 0.5) * 255);
        nImg.data[idx + 1] = Math.floor((dy * inv * 0.5 + 0.5) * 255);
        nImg.data[idx + 2] = Math.floor((dz * inv * 0.5 + 0.5) * 255);
        nImg.data[idx + 3] = 255;

        // 粗糙度空间变化：叶脉及微尘处更哑光粗糙 (190~240)，叶肉处适度亚光 (145~185)，彻底告别均匀塑料高光
        var hC = srcData[idx];
        var rv = Math.min(248, Math.max(138, Math.floor(155 + (hC - 112) * 0.45 + (rngR() - 0.5) * 42)));
        rImg.data[idx] = rImg.data[idx + 1] = rImg.data[idx + 2] = rv;
        rImg.data[idx + 3] = 255;
      }
    }
    gn.putImageData(nImg, 0, 0);
    gr.putImageData(rImg, 0, 0);

    var normalTex = new THREE.CanvasTexture(cNorm); normalTex.anisotropy = 8;
    var roughTex = new THREE.CanvasTexture(cRough); roughTex.anisotropy = 8;
    var aoTex = new THREE.CanvasTexture(cAo);

    // 甜菜叶片真实波状/微锯齿叶缘 Alpha 遮罩贴图（彻底消除程序化曲面的直边折纸感）
    var cAlpha = document.createElement('canvas'); cAlpha.width = sz; cAlpha.height = sz;
    var gal = cAlpha.getContext('2d');
    gal.fillStyle = '#000000'; gal.fillRect(0, 0, sz, sz);
    gal.fillStyle = '#ffffff';
    // 叶柄与叶面沿 UV 横向边界做波浪锯齿裁剪（UV 中 x=0/1 对应叶片左右边缘，y=0..1 对应叶基到叶尖）
    gal.beginPath();
    var stepsA = 64;
    for (var ai = 0; ai <= stepsA; ai++) {
      var vNorm = ai / stepsA;
      var pyA = (1.0 - vNorm) * sz;
      var edgeInset = vNorm < 0.24 ? 0.08 : (0.04 + Math.sin(vNorm * 38.0) * 0.032 + Math.cos(vNorm * 62.0) * 0.014);
      var pxL = Math.max(2, edgeInset * sz * 0.5);
      if (ai === 0) gal.moveTo(pxL, pyA); else gal.lineTo(pxL, pyA);
    }
    for (var aj = stepsA; aj >= 0; aj--) {
      var vNormR = aj / stepsA;
      var pyAR = (1.0 - vNormR) * sz;
      var edgeInsetR = vNormR < 0.24 ? 0.08 : (0.04 + Math.sin(vNormR * 38.0 + 1.1) * 0.032 + Math.cos(vNormR * 62.0 + 0.7) * 0.014);
      var pxR = Math.min(sz - 2, sz - edgeInsetR * sz * 0.5);
      gal.lineTo(pxR, pyAR);
    }
    gal.closePath();
    gal.fill();
    var alphaTex = new THREE.CanvasTexture(cAlpha);
    alphaTex.anisotropy = 8;

    return { map: mapTex, mapChlorosis: mapChlTex, bump: bumpTex, normal: normalTex, rough: roughTex, ao: aoTex, alpha: alphaTex };
  }

  // 通用低频噪点粗糙度与微凹凸贴图（赋予花盆、金属架、柜体真实表面微瑕，消除均匀塑料感）
  function makeMicroSurfaceMaps() {
    var sz = 128;
    var cR = document.createElement('canvas'); cR.width = sz; cR.height = sz;
    var gr = cR.getContext('2d');
    var cB = document.createElement('canvas'); cB.width = sz; cB.height = sz;
    var gb = cB.getContext('2d');
    var rng = mulberry32(9901);
    for (var y = 0; y < sz; y += 2) {
      for (var x = 0; x < sz; x += 2) {
        var lf = Math.sin(x * 0.09) * Math.cos(y * 0.09) * 22;
        var rv = Math.min(245, Math.max(120, Math.floor(178 + lf + (rng() - 0.5) * 44)));
        gr.fillStyle = 'rgb(' + rv + ',' + rv + ',' + rv + ')';
        gr.fillRect(x, y, 2, 2);
        var bv = Math.min(255, Math.max(0, Math.floor(128 + lf * 1.4 + (rng() - 0.5) * 32)));
        gb.fillStyle = 'rgb(' + bv + ',' + bv + ',' + bv + ')';
        gb.fillRect(x, y, 2, 2);
      }
    }
    var rTex = new THREE.CanvasTexture(cR); rTex.wrapS = rTex.wrapT = THREE.RepeatWrapping; rTex.repeat.set(4, 4);
    var bTex = new THREE.CanvasTexture(cB); bTex.wrapS = bTex.wrapT = THREE.RepeatWrapping; bTex.repeat.set(4, 4);
    return { rough: rTex, bump: bTex, normal: heightCanvasToNormalTex(cB, 2.2, 4, 4) };
  }

  // 带 3 段平滑斜切圆角倒角 (Bevel) + 可选 3D FBM 表面微位移的立体构件生成器
  function createBeveledBox(w, h, d, bev, noiseAmp, seed) {
    var b = Math.min(bev || 0.013, w * 0.22, h * 0.22, d * 0.22);
    var iw = w - b * 2, ih = h - b * 2;
    var shape = new THREE.Shape();
    shape.moveTo(-iw / 2, -ih / 2);
    shape.lineTo(iw / 2, -ih / 2);
    shape.lineTo(iw / 2, ih / 2);
    shape.lineTo(-iw / 2, ih / 2);
    shape.closePath();
    var geo = new THREE.ExtrudeGeometry(shape, {
      depth: Math.max(0.002, d - b * 2),
      bevelEnabled: true,
      bevelSegments: 3,
      steps: noiseAmp ? 4 : 1,
      bevelSize: b,
      bevelThickness: b
    });
    geo.center();
    return prepareAndDisplaceGeometry(geo, noiseAmp || 0, noiseAmp || 0, noiseAmp || 0, 3.5, seed || 202);
  }

  // 高细分 3D 凹凸起伏营养土几何体 (Subdivision + 3D FBM Vertex Displacement)
  function createDisplacedSoilGeometry(radius, height, seed) {
    var geo = new THREE.CylinderGeometry(radius, radius * 0.96, height, 32, 4);
    var pos = geo.attributes.position;
    for (var i = 0; i < pos.count; i++) {
      var vx = pos.getX(i), vy = pos.getY(i), vz = pos.getZ(i);
      var r = Math.hypot(vx, vz) / Math.max(0.001, radius);
      if (vy > 0) {
        var mound = Math.cos(Math.min(1, r) * Math.PI * 0.5) * 0.011;
        var clod = fbm3D(vx * 38, vy * 38, vz * 38, seed || 505) * 0.0052;
        vy += mound + clod;
      }
      var edgeWarp = fbm3D(vx * 22, vy * 22, vz * 22, (seed || 505) + 19) * 0.002;
      pos.setXYZ(i, vx + edgeWarp, vy, vz + edgeWarp);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    if (geo.attributes.uv && !geo.attributes.uv2) geo.setAttribute('uv2', geo.attributes.uv.clone());
    return geo;
  }

  // 甜菜膨大块根根冠几何体 (Beta vulgaris Root Crown，高细分 + 3D 环纹与有机疙瘩位移)
  function createSugarBeetRootCrownGeometry(seed) {
    var geo = new THREE.SphereGeometry(0.035, 22, 14);
    var pos = geo.attributes.position;
    for (var i = 0; i < pos.count; i++) {
      var vx = pos.getX(i), vy = pos.getY(i), vz = pos.getZ(i);
      var ring = Math.sin(vy * 140) * 0.0016;
      var bump = fbm3D(vx * 45, vy * 45, vz * 45, seed || 808) * 0.0035;
      var taper = vy < 0 ? (1.0 + vy * 8.5) : (1.0 - vy * 4.2);
      taper = Math.max(0.35, taper);
      pos.setXYZ(i, vx * taper + bump + ring, vy * 1.15, vz * taper + bump + ring);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    if (geo.attributes.uv && !geo.attributes.uv2) geo.setAttribute('uv2', geo.attributes.uv.clone());
    return geo;
  }

  // 带物理受压微形变的高细分栽培盆与圆角倒角盆沿
  function createDeformedPotGeometry(seed) {
    var geo = new THREE.CylinderGeometry(0.16, 0.125, 0.22, 32, 6);
    return prepareAndDisplaceGeometry(geo, 0.0022, 0.0008, 0.0022, 14.0, seed || 611);
  }
  function createBeveledPotRimGeometry(seed) {
    var geo = new THREE.TorusGeometry(0.162, 0.013, 12, 32);
    geo.rotateX(Math.PI / 2);
    return prepareAndDisplaceGeometry(geo, 0.0016, 0.0010, 0.0016, 16.0, seed || 619);
  }

  function makeVolumetricMistTex() {
    var sz = 256;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(sz / 2, sz / 2, 10, sz / 2, sz / 2, sz * 0.48);
    grd.addColorStop(0.0, 'rgba(225,240,250,0.14)');
    grd.addColorStop(0.55, 'rgba(215,235,248,0.06)');
    grd.addColorStop(1.0, 'rgba(215,235,248,0.0)');
    g.fillStyle = grd; g.fillRect(0, 0, sz, sz);
    return new THREE.CanvasTexture(c);
  }

  function makeGroundMatTex() {
    var sz = 512;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    g.fillStyle = '#dce2e5'; g.fillRect(0, 0, sz, sz);
    var rng = mulberry32(101);
    // 编织防草地布细密经纬纹
    g.strokeStyle = 'rgba(176,186,192,0.42)'; g.lineWidth = 1;
    for (var i = 0; i < sz; i += 6) {
      g.beginPath(); g.moveTo(i, 0); g.lineTo(i, sz); g.stroke();
      g.beginPath(); g.moveTo(0, i); g.lineTo(sz, i); g.stroke();
    }
    // 自然灰尘微污渍斑点，打破机械重复感
    for (var j = 0; j < 2600; j++) {
      var v = 195 + Math.floor(rng() * 32);
      g.fillStyle = 'rgba(' + v + ',' + (v - 2) + ',' + (v - 6) + ',0.22)';
      g.fillRect(rng() * sz, rng() * sz, 2 + rng() * 4, 2 + rng() * 4);
    }
    // 设施温室防草布标志性暗绿定位线
    g.strokeStyle = 'rgba(46,138,88,0.42)'; g.lineWidth = 2.5;
    g.beginPath(); g.moveTo(0, sz / 2); g.lineTo(sz, sz / 2); g.stroke();
    g.beginPath(); g.moveTo(sz / 2, 0); g.lineTo(sz / 2, sz); g.stroke();

    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(7, 4.5);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    return tex;
  }

  function makeAbsTrayTex() {
    var sz = 256;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    g.fillStyle = '#2a3035'; g.fillRect(0, 0, sz, sz);
    // 潮汐苗床 ABS 托盘导水凹凸肋条
    for (var x = 0; x < sz; x += 16) {
      g.fillStyle = '#1b2024'; g.fillRect(x, 0, 5, sz);
      g.fillStyle = '#384047'; g.fillRect(x + 5, 0, 2, sz);
    }
    for (var y = 0; y < sz; y += 64) {
      g.fillStyle = '#171b1f'; g.fillRect(0, y, sz, 4);
    }
    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(16, 2);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    return tex;
  }

  function makeGrassFieldTex() {
    var sz = 512;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    g.fillStyle = '#4e6f3a'; g.fillRect(0, 0, sz, sz);
    var rng = mulberry32(303);
    // 大尺度草地深浅色块
    for (var k = 0; k < 180; k++) {
      var gx = rng() * sz, gy = rng() * sz, rad = 14 + rng() * 34;
      g.fillStyle = rng() > 0.5 ? 'rgba(66,98,46,0.22)' : 'rgba(96,124,62,0.18)';
      g.beginPath(); g.arc(gx, gy, rad, 0, Math.PI * 2); g.fill();
    }
    // 细腻草叶纤维
    for (var i = 0; i < 14000; i++) {
      var r = 60 + Math.floor(rng() * 44), gr = 92 + Math.floor(rng() * 54), b = 42 + Math.floor(rng() * 28);
      g.fillStyle = 'rgb(' + r + ',' + gr + ',' + b + ')';
      g.fillRect(rng() * sz, rng() * sz, 1.8, 3.5);
    }
    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(28, 28);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    return tex;
  }

  function makeConcreteTex() {
    var sz = 256;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    g.fillStyle = '#a2aaaf'; g.fillRect(0, 0, sz, sz);
    var rng = mulberry32(404);
    for (var i = 0; i < 6000; i++) {
      var v = 135 + Math.floor(rng() * 52);
      g.fillStyle = 'rgb(' + v + ',' + (v + 2) + ',' + (v + 3) + ')';
      g.fillRect(rng() * sz, rng() * sz, 2, 2);
    }
    // 底部返潮略深渐变
    var damp = g.createLinearGradient(0, sz, 0, sz * 0.6);
    damp.addColorStop(0, 'rgba(65,72,76,0.35)');
    damp.addColorStop(1, 'rgba(65,72,76,0)');
    g.fillStyle = damp; g.fillRect(0, 0, sz, sz);
    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(6, 1);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  function makeSoilTex() {
    var sz = 512;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    var cH = document.createElement('canvas'); cH.width = sz; cH.height = sz;
    var gh = cH.getContext('2d');
    var cR = document.createElement('canvas'); cR.width = sz; cR.height = sz;
    var gr = cR.getContext('2d');

    g.fillStyle = '#2d1e14'; g.fillRect(0, 0, sz, sz);
    gh.fillStyle = '#707070'; gh.fillRect(0, 0, sz, sz);
    gr.fillStyle = '#eaeaea'; gr.fillRect(0, 0, sz, sz);

    var rng = mulberry32(505);
    // 1. 干湿团粒基质色块起伏
    for (var k = 0; k < 220; k++) {
      var sx = rng() * sz, sy = rng() * sz, rad = 8 + rng() * 24;
      var wet = rng() > 0.48;
      g.fillStyle = wet ? 'rgba(28,18,10,0.34)' : 'rgba(72,49,32,0.28)';
      g.beginPath(); g.arc(sx, sy, rad, 0, Math.PI * 2); g.fill();
      gh.fillStyle = wet ? 'rgba(92,92,92,0.32)' : 'rgba(155,155,155,0.32)';
      gh.beginPath(); gh.arc(sx, sy, rad, 0, Math.PI * 2); gh.fill();
      gr.fillStyle = wet ? 'rgba(168,168,168,0.4)' : 'rgba(244,244,244,0.4)';
      gr.beginPath(); gr.arc(sx, sy, rad, 0, Math.PI * 2); gr.fill();
    }
    // 2. 泥炭细团粒 + 白色珍珠岩 + 金黄蛭石 + 椰糠粗纤维
    for (var i = 0; i < 9500; i++) {
      var p = rng();
      var px = rng() * sz, py = rng() * sz;
      if (p < 0.045) {
        g.fillStyle = '#e6dfd3'; // 白色珍珠岩颗粒
        g.fillRect(px, py, 3.2, 2.8);
        gh.fillStyle = '#e8e8e8';
        gh.fillRect(px, py, 3.2, 2.8);
      } else if (p < 0.075) {
        g.fillStyle = '#b48b48'; // 金黄蛭石片
        g.fillRect(px, py, 2.8, 2.0);
        gh.fillStyle = '#d4d4d4';
        gh.fillRect(px, py, 2.8, 2.0);
      } else if (p < 0.14) {
        g.fillStyle = '#624129'; // 椰糠粗纤维丝
        g.fillRect(px, py, 5.2, 1.5);
        gh.fillStyle = '#c8c8c8';
        gh.fillRect(px, py, 5.2, 1.5);
      } else {
        var r = 30 + Math.floor(rng() * 44), grn = 20 + Math.floor(rng() * 28), b = 12 + Math.floor(rng() * 16);
        g.fillStyle = 'rgb(' + r + ',' + grn + ',' + b + ')';
        g.fillRect(px, py, 2.6, 2.6);
        var hv = 95 + Math.floor(rng() * 90);
        gh.fillStyle = 'rgb(' + hv + ',' + hv + ',' + hv + ')';
        gh.fillRect(px, py, 2.6, 2.6);
      }
    }
    // 根际中心微湿深色圈
    var rWet = g.createRadialGradient(sz / 2, sz / 2, 10, sz / 2, sz / 2, sz * 0.38);
    rWet.addColorStop(0, 'rgba(18,11,6,0.45)');
    rWet.addColorStop(1, 'rgba(18,11,6,0)');
    g.fillStyle = rWet; g.fillRect(0, 0, sz, sz);

    var tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    var rTex = new THREE.CanvasTexture(cR);
    return { map: tex, normal: heightCanvasToNormalTex(cH, 3.6, 1, 1), rough: rTex };
  }

  // 真实户外农田土地 PBR 纹理组（深棕壤土团粒 + 砂砾 + 干湿色块 + 枯草根茬）
  function makeRealisticFieldSoilSet() {
    var sz = 512;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    var cH = document.createElement('canvas'); cH.width = sz; cH.height = sz;
    var gh = cH.getContext('2d');
    var cR = document.createElement('canvas'); cR.width = sz; cR.height = sz;
    var gr = cR.getContext('2d');

    g.fillStyle = '#4f3b2b'; g.fillRect(0, 0, sz, sz);
    gh.fillStyle = '#808080'; gh.fillRect(0, 0, sz, sz);
    gr.fillStyle = '#dcdcdc'; gr.fillRect(0, 0, sz, sz);

    var rng = mulberry32(202509);
    // 1. 大尺度干湿土壤色斑与土垄肌理
    for (var k = 0; k < 280; k++) {
      var x = rng() * sz, y = rng() * sz, rad = 16 + rng() * 48;
      var moist = rng() > 0.45;
      g.fillStyle = moist ? 'rgba(54,39,27,0.28)' : 'rgba(104,81,60,0.22)';
      g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
      gh.fillStyle = moist ? 'rgba(102,102,102,0.25)' : 'rgba(154,154,154,0.25)';
      gh.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
      gr.fillStyle = moist ? 'rgba(176,176,176,0.3)' : 'rgba(242,242,242,0.3)';
      gr.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); gr.fill();
    }
    // 2. 土块颗粒、细砂砾、碎石与零星枯草茬
    for (var i = 0; i < 16000; i++) {
      var px = rng() * sz, py = rng() * sz;
      var p = rng();
      if (p < 0.03) {
        // 小碎石砂砾
        var sv = 115 + Math.floor(rng() * 55);
        g.fillStyle = 'rgb(' + sv + ',' + (sv - 6) + ',' + (sv - 14) + ')';
        g.fillRect(px, py, 2.5 + rng() * 2.5, 2.0 + rng() * 2.0);
        gh.fillStyle = '#dcdcdc';
        gh.fillRect(px, py, 3, 2.5);
      } else if (p < 0.055) {
        // 细碎有机枯草茬
        g.strokeStyle = 'rgba(136,118,74,0.65)';
        g.lineWidth = 1.1;
        g.beginPath(); g.moveTo(px, py); g.lineTo(px + (rng() - 0.5) * 9, py + (rng() - 0.5) * 9); g.stroke();
      } else {
        var r = 64 + Math.floor(rng() * 48), grn = 48 + Math.floor(rng() * 36), b = 34 + Math.floor(rng() * 26);
        g.fillStyle = 'rgb(' + r + ',' + grn + ',' + b + ')';
        g.fillRect(px, py, 2.2 + rng() * 2.4, 2.2 + rng() * 2.4);
        var hv = 90 + Math.floor(rng() * 88);
        gh.fillStyle = 'rgb(' + hv + ',' + hv + ',' + hv + ')';
        gh.fillRect(px, py, 2.5, 2.5);
      }
    }

    function repTex(cv, isSRGB) {
      var t = new THREE.CanvasTexture(cv);
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(3.5, 2.6);
      t.anisotropy = 8;
      if (isSRGB) t.encoding = THREE.sRGBEncoding;
      return t;
    }
    return {
      map: repTex(c, true),
      normalMap: heightCanvasToNormalTex(cH, 3.2, 3.5, 2.6),
      roughnessMap: repTex(cR, false)
    };
  }

  function makeCoolingPadTex() {
    var sz = 256;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    var cH = document.createElement('canvas'); cH.width = sz; cH.height = sz;
    var gh = cH.getContext('2d');

    // 真实温室 7090 棕褐色/深灰褐蜂窝波纹湿帘纸 + 瓦楞法线凹凸
    g.fillStyle = '#6c4e31'; g.fillRect(0, 0, sz, sz);
    gh.fillStyle = '#787878'; gh.fillRect(0, 0, sz, sz);
    for (var y = -sz; y < sz * 2; y += 7) {
      g.strokeStyle = 'rgba(38,24,12,0.72)'; g.lineWidth = 2.2;
      g.beginPath(); g.moveTo(0, y); g.lineTo(sz, y + 72); g.stroke();
      g.strokeStyle = 'rgba(154,118,76,0.48)'; g.lineWidth = 1.6;
      g.beginPath(); g.moveTo(0, y + 3); g.lineTo(sz, y - 66); g.stroke();

      gh.strokeStyle = 'rgba(42,42,42,0.8)'; gh.lineWidth = 2.2;
      gh.beginPath(); gh.moveTo(0, y); gh.lineTo(sz, y + 72); gh.stroke();
      gh.strokeStyle = 'rgba(210,210,210,0.8)'; gh.lineWidth = 1.6;
      gh.beginPath(); gh.moveTo(0, y + 3); gh.lineTo(sz, y - 66); gh.stroke();
    }
    var wet = g.createLinearGradient(0, sz, 0, sz * 0.50);
    wet.addColorStop(0, 'rgba(22,14,6,0.52)');
    wet.addColorStop(1, 'rgba(22,14,6,0)');
    g.fillStyle = wet; g.fillRect(0, 0, sz, sz);

    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3, 2);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    return { map: tex, normal: heightCanvasToNormalTex(cH, 3.0, 3, 2) };
  }

  function makeContactShadowTex() {
    var sz = 128;
    var c = document.createElement('canvas'); c.width = sz; c.height = sz;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(sz / 2, sz / 2, sz * 0.12, sz / 2, sz / 2, sz * 0.5);
    grd.addColorStop(0.0, 'rgba(0,0,0,0.52)');
    grd.addColorStop(0.55, 'rgba(0,0,0,0.22)');
    grd.addColorStop(1.0, 'rgba(0,0,0,0.0)');
    g.fillStyle = grd; g.fillRect(0, 0, sz, sz);
    return new THREE.CanvasTexture(c);
  }

  function makeSunShaftTex() {
    var c = document.createElement('canvas'); c.width = 128; c.height = 256;
    var g = c.getContext('2d');
    var gx = g.createLinearGradient(0, 0, 128, 0);
    gx.addColorStop(0, 'rgba(255,246,214,0)');
    gx.addColorStop(0.5, 'rgba(255,246,214,0.22)');
    gx.addColorStop(1, 'rgba(255,246,214,0)');
    g.fillStyle = gx; g.fillRect(0, 0, 128, 256);
    g.globalCompositeOperation = 'destination-in';
    var gy = g.createLinearGradient(0, 0, 0, 256);
    gy.addColorStop(0, 'rgba(255,255,255,0.95)');
    gy.addColorStop(0.75, 'rgba(255,255,255,0.35)');
    gy.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gy; g.fillRect(0, 0, 128, 256);
    return new THREE.CanvasTexture(c);
  }

  /* ==================== 2. 高细分三维参数化甜菜叶片几何（20×28 细分 + 3D FBM 位移） ==================== */

  function createParametricBeetLeafGeometry(variantSeed) {
    var segU = 20, segV = 28;
    var vSeed = variantSeed || 1;
    var positions = [];
    var uvs = [];
    var indices = [];

    for (var iv = 0; iv <= segV; iv++) {
      var v = iv / segV; // 0 = 叶柄底端, 1 = 叶尖
      var halfW;
      if (v < 0.24) {
        halfW = 0.011 + 0.010 * (v / 0.24);
      } else {
        var tv = (v - 0.24) / 0.76;
        halfW = 0.021 + 0.128 * Math.pow(Math.sin(Math.pow(tv, 0.74) * Math.PI), 0.84);
      }

      var lenY = v * 0.445;
      var archZ = Math.pow(v, 1.85) * (0.235 + (vSeed % 3) * 0.012);

      for (var iu = 0; iu <= segU; iu++) {
        var uNorm = iu / segU; // 0..1
        var s = uNorm * 2 - 1; // -1..1
        var x = s * halfW;
        var absS = Math.abs(s);

        // 横向解剖截面：叶柄深 U 槽 -> 叶面中脉隆起 + 侧脉间叶肉泡状 3D 鼓起 (Bullate Displacement) + 叶缘双频褶皱 + 3D FBM 位移
        var cup = (v < 0.24)
          ? (absS * absS * 0.014 - (1 - absS) * 0.004)
          : (absS * 0.019 - absS * absS * 0.013);
        var veinBullate = (v > 0.24)
          ? Math.sin(v * 34.0 - absS * 7.5) * Math.sin(absS * Math.PI) * 0.0055
          : 0;
        var ruffle = (v > 0.25)
          ? (Math.sin(v * 19.5 + s * 3.6 + vSeed * 1.3) * 0.019 + Math.cos(v * 31.0 - s * 5.2) * 0.007) * Math.pow(absS, 1.5)
          : 0;
        var fbmDisp = fbm3D(s * 3.8, v * 4.5, vSeed * 2.1, 700 + vSeed * 31) * 0.0085 * (0.3 + 0.7 * v);
        var asymTwist = s * Math.pow(v, 1.5) * (0.015 + (vSeed % 2 ? 0.005 : -0.004)) + Math.sin(v * 7.5 + s * 2.1) * 0.007;
        var xAssym = x + Math.pow(v, 2.1) * (0.013 * (vSeed % 2 ? 1 : -0.8)) + fbmDisp * 0.35;
        var z = archZ + cup + veinBullate + ruffle + asymTwist + fbmDisp;
        var y = lenY - Math.pow(v, 2.8) * 0.062 + fbmDisp * 0.25;

        positions.push(xAssym, y, z);
        uvs.push(uNorm, v);
      }
    }

    var rowLen = segU + 1;
    for (var jv = 0; jv < segV; jv++) {
      for (var ju = 0; ju < segU; ju++) {
        var a = jv * rowLen + ju;
        var b = a + 1;
        var c = a + rowLen;
        var d = c + 1;
        indices.push(a, b, d, a, d, c);
      }
    }

    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setAttribute('uv2', new THREE.Float32BufferAttribute(uvs, 2)); // 供 aoMap 环境光遮蔽使用
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }

  /* ==================== 3. 主初始化入口 ==================== */

  global.initGreenhouse3D = function (opts) {
    var D = opts.data;
    var S = D.summary, U = D.units;
    var LV = opts.LV || ['#3ddc84', '#ffb020', '#ff7a45', '#ff4d67'];
    var LVN = opts.LVN || ['健康', '早期轻度', '中度', '重度'];

    var canvas = document.getElementById('scene3d');
    var wrapEl = document.getElementById('scene-wrap');
    var labelsEl = document.getElementById('labels3d');
    var tip3d = document.getElementById('tip3d');
    var legendBar = document.getElementById('legend-bar');
    var rowsEl = document.getElementById('rows');

    var mode = 'grade';
    var sceneTheme = 'real'; // 默认采用白天日光物理环境，真实还原温室采光与土壤质感
    // 拓宽床间距离至 2.36m，配合 1.04m 标准苗床宽度，留出两条 1.32m 宽的标准科研人员巡检与表型作业过道
    var BED_Z = { W1: -2.36, W2: 0.0, W3: 2.36 };
    var BED_N = {
      W1: 'W1 对照床 · 持续供水',
      W2: 'W2 断水床 A · 同期断水',
      W3: 'W3 断水床 B · 同期断水'
    };
    var BED_COL = { W1: '#38bdf8', W2: '#fbbf24', W3: '#fb7185' };
    var BED_PIPE_HEX = { W1: 0x38bdf8, W2: 0xfbbf24, W3: 0xfb7185 };
    var allCodes = Array.from(new Set(U.map(function (u) { return u.code; }))).sort();

    function residHex(r) {
      // 标准热红外伪彩分段（0℃ -> +8℃）：冷蓝 -> 青绿 -> 明黄 -> 亮橙 -> 炽红
      var t = Math.max(0, Math.min(1, r / 8.0));
      if (t < 0.22) return 0x2563eb;
      if (t < 0.42) return 0x06b6d4;
      if (t < 0.62) return 0xeab308;
      if (t < 0.80) return 0xf97316;
      return 0xef4444;
    }
    function gradeHex(g, isCtrl) {
      if (isCtrl) return 0x38bdf8;
      return [0x3ddc84, 0xffb020, 0xff7a45, 0xff4d67][g] || 0x3ddc84;
    }
    // 生理级叶片底色：胁迫等级模式展现油绿/黄化；残差温度模式切换为红外伪彩温度场（对照床呈现冷蓝基线）
    function leafColorHex(u, curMode) {
      if (curMode === 'resid') return u.isCtrl ? 0x1e6091 : residHex(u.resid * (global._replayScale != null ? global._replayScale : 1));
      if (u.isCtrl) return 0x1c6326;
      if (u.bed === 'W2') {
        return [0x24682a, 0x4b7726, 0x6c7c24, 0x867822][u.grade] || 0x4b7726;
      }
      return [0x2c6828, 0x5c7424, 0x827422, 0x986e24][u.grade] || 0x827422;
    }

    // WebGL 高画质渲染器
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.80;
    renderer.outputEncoding = THREE.sRGBEncoding;

    var scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x04111b, 0.024);

    // 1024×512 高清蓝天白云太阳穹顶 + PMREM 环境反射贴图
    var skyTex = (function createSkyTex() {
      var c = document.createElement('canvas'); c.width = 1024; c.height = 512;
      var g = c.getContext('2d');
      var grd = g.createLinearGradient(0, 0, 0, 512);
      grd.addColorStop(0.00, '#3a78bc');
      grd.addColorStop(0.32, '#7db4e0');
      grd.addColorStop(0.49, '#dcebf5');
      grd.addColorStop(0.53, '#cfd8dc');
      grd.addColorStop(1.00, '#6e7b74');
      g.fillStyle = grd; g.fillRect(0, 0, 1024, 512);

      // 柔和太阳光晕
      var sunG = g.createRadialGradient(780, 130, 8, 780, 130, 145);
      sunG.addColorStop(0, 'rgba(255,253,235,0.95)');
      sunG.addColorStop(0.25, 'rgba(255,244,205,0.45)');
      sunG.addColorStop(1, 'rgba(255,244,205,0)');
      g.fillStyle = sunG; g.fillRect(620, 0, 320, 280);

      // 多层积云
      var rng = mulberry32(909);
      for (var i = 0; i < 48; i++) {
        var cx = rng() * 1024, cy = 70 + rng() * 140;
        var rw = 40 + rng() * 85, rh = 10 + rng() * 18;
        g.fillStyle = 'rgba(255,255,255,' + (0.18 + rng() * 0.25).toFixed(2) + ')';
        g.beginPath(); g.ellipse(cx, cy, rw, rh, 0, 0, Math.PI * 2); g.fill();
      }
      var t = new THREE.CanvasTexture(c);
      t.encoding = THREE.sRGBEncoding;
      return t;
    })();

    var skyDome = new THREE.Mesh(
      new THREE.SphereGeometry(85, 32, 20),
      new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, fog: false })
    );
    skyDome.visible = false; // 暗夜科研模式默认关闭室外白亮天空球，融入深青色仪表盘
    scene.add(skyDome);

    // 程序化 HDRI 全局照明 (PMREM GI) + 异步升级本地 Poly Haven 真实 1K .hdr 环境贴图
    (function buildHDRIEnvIBL() {
      var pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      var envScene = new THREE.Scene();
      envScene.add(new THREE.Mesh(
        new THREE.SphereGeometry(22, 24, 16),
        new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide })
      ));
      var sunDisk = new THREE.Mesh(
        new THREE.SphereGeometry(1.8, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xfff8e7 })
      );
      sunDisk.position.set(14, 12, 8);
      envScene.add(sunDisk);
      var bounceFloor = new THREE.Mesh(
        new THREE.CircleGeometry(18, 24),
        new THREE.MeshBasicMaterial({ color: 0x48443c, side: THREE.DoubleSide })
      );
      bounceFloor.rotation.x = -Math.PI / 2;
      bounceFloor.position.y = -2.0;
      envScene.add(bounceFloor);
      scene.environment = pmrem.fromScene(envScene, 0.035).texture;

      if (typeof THREE.RGBELoader === 'function') {
        new THREE.RGBELoader().setDataType(THREE.UnsignedByteType).load('assets/sky_1k.hdr', function (hdrTex) {
          var envRT = pmrem.fromEquirectangular(hdrTex);
          scene.environment = envRT.texture;
          hdrTex.dispose();
          pmrem.dispose();
        }, undefined, function () {
          pmrem.dispose();
        });
      } else {
        pmrem.dispose();
      }
    })();

    // 50mm 真实摄影标准镜头视角 (35° FOV)，降低俯视角以抬高地平线、压缩上方空旷土地并完整展示温室主体
    var DEFAULT_CAM_POS = [-4.65, 2.38, 5.20];
    var DEFAULT_LOOK_AT = [0.15, 1.15, 0.0];
    var camera = new THREE.PerspectiveCamera(35, 1.6, 0.15, 180);
    camera.position.set(DEFAULT_CAM_POS[0], DEFAULT_CAM_POS[1], DEFAULT_CAM_POS[2]);

    var controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.minDistance = 2.0;
    controls.maxDistance = 22;
    controls.target.set(DEFAULT_LOOK_AT[0], DEFAULT_LOOK_AT[1], DEFAULT_LOOK_AT[2]);
    controls.update();

    // 电影级冷暖对比三点布光系统（暗夜科研舞台基底：冷青轮廓光 + 柔和暖调主光 + 宽半影软阴影）
    var hemiLight = new THREE.HemisphereLight(0x7be8d8, 0x0b1c26, 0.36);
    scene.add(hemiLight);

    var dirLight = new THREE.DirectionalLight(0xfff0d4, 1.12);
    dirLight.position.set(10.5, 10.2, 6.2);
    dirLight.castShadow = true;
    var maxShadowSz = Math.min(4096, renderer.capabilities.maxTextureSize || 2048);
    dirLight.shadow.mapSize.width = maxShadowSz;
    dirLight.shadow.mapSize.height = maxShadowSz;
    dirLight.shadow.radius = 3.8;
    dirLight.shadow.bias = -0.00025;
    dirLight.shadow.camera.near = 2;
    dirLight.shadow.camera.far = 28;
    var dRange = 7.2; // 严密收紧阴影相机视锥刚好包住温室主体，最大化阴影贴图精度
    dirLight.shadow.camera.left = -dRange;
    dirLight.shadow.camera.right = dRange;
    dirLight.shadow.camera.top = dRange;
    dirLight.shadow.camera.bottom = -dRange;
    scene.add(dirLight);

    var coolFillLight = new THREE.DirectionalLight(0x5eb8ff, 0.42);
    coolFillLight.position.set(-9.5, 7.2, 5.5);
    scene.add(coolFillLight);

    var rimLight = new THREE.DirectionalLight(0x35e0c0, 0.96);
    rimLight.position.set(-11, 6.2, -9.5);
    scene.add(rimLight);

    var warmBounceRim = new THREE.DirectionalLight(0xffb36b, 0.25);
    warmBounceRim.position.set(8, 1.2, -8);
    scene.add(warmBounceRim);

    // 批量生成全通道 PBR 贴图与边缘磨损/表面微瑕贴图组
    var leafTexSet = makeBeetLeafTextures();
    var glassReflTex = makeGlassReflectionTex();
    var microSurf = makeMicroSurfaceMaps();
    var groundMatTex = makeGroundMatTex();
    var absTrayTex = makeAbsTrayTex();
    var grassTex = makeGrassFieldTex();
    var concreteTex = makeConcreteTex();
    var soilTexSet = makeSoilTex();
    var fieldSoilSet = makeRealisticFieldSoilSet();
    var padTex = makeCoolingPadTex();
    var shadowTex = makeContactShadowTex();
    var shaftTex = makeSunShaftTex();
    var mistTex = makeVolumetricMistTex();

    // 工业构件专属全通道 PBR 材质组
    var galvSteelPBR = makeWeatheredPBRSet({
      baseRGB: [194, 204, 211], wearRGB: [238, 243, 247], dirtRGB: [118, 124, 128],
      roughness: 0.44, normalStrength: 2.3, repeatX: 3, repeatY: 3, seed: 1101
    });
    var darkSteelPBR = makeWeatheredPBRSet({
      baseRGB: [56, 70, 82], wearRGB: [130, 144, 156], dirtRGB: [32, 38, 44],
      roughness: 0.54, normalStrength: 2.5, repeatX: 3, repeatY: 3, seed: 1202
    });
    var alumFramePBR = makeWeatheredPBRSet({
      baseRGB: [182, 192, 199], wearRGB: [232, 238, 242], dirtRGB: [108, 114, 118],
      roughness: 0.42, normalStrength: 2.1, repeatX: 4, repeatY: 2, seed: 1303
    });
    var pvcPBR = makeWeatheredPBRSet({
      baseRGB: [224, 230, 235], wearRGB: [246, 248, 250], dirtRGB: [152, 148, 138],
      roughness: 0.56, normalStrength: 1.7, repeatX: 3, repeatY: 1, seed: 1404
    });
    var potPlasticPBR = makeWeatheredPBRSet({
      baseRGB: [40, 44, 49], wearRGB: [88, 94, 102], dirtRGB: [112, 98, 82],
      roughness: 0.80, normalStrength: 2.4, repeatX: 2, repeatY: 2, seed: 1505
    });

    var archMat = new THREE.MeshStandardMaterial({
      color: 0xd8e1e8,
      map: galvSteelPBR.map, roughnessMap: galvSteelPBR.roughnessMap,
      normalMap: galvSteelPBR.normalMap, normalScale: new THREE.Vector2(0.45, 0.45),
      aoMap: galvSteelPBR.aoMap, aoMapIntensity: 0.45,
      roughness: 0.30, metalness: 0.52, envMapIntensity: 1.25
    });
    var darkSteelMat = new THREE.MeshStandardMaterial({
      map: darkSteelPBR.map, roughnessMap: darkSteelPBR.roughnessMap,
      normalMap: darkSteelPBR.normalMap, normalScale: new THREE.Vector2(0.6, 0.6),
      aoMap: darkSteelPBR.aoMap, aoMapIntensity: 0.6,
      roughness: 0.44, metalness: 0.68, envMapIntensity: 0.58
    });
    var curbMat = new THREE.MeshStandardMaterial({
      map: concreteTex, bumpMap: concreteTex, bumpScale: 0.015,
      normalMap: microSurf.normal, normalScale: new THREE.Vector2(0.65, 0.65),
      roughnessMap: microSurf.rough, aoMap: galvSteelPBR.aoMap, aoMapIntensity: 0.5,
      color: 0xbec5c9, roughness: 0.92, metalness: 0.02, envMapIntensity: 0.14
    });
    var pvcMat = new THREE.MeshStandardMaterial({
      map: pvcPBR.map, roughnessMap: pvcPBR.roughnessMap,
      normalMap: pvcPBR.normalMap, normalScale: new THREE.Vector2(0.4, 0.4),
      aoMap: pvcPBR.aoMap, aoMapIntensity: 0.45,
      roughness: 0.58, metalness: 0.04, envMapIntensity: 0.20
    });
    // 温室覆盖材质（PO 漫反射膜 / 散射玻璃感）：清透天青反光 + 清晰表面高光，避免“纯透明像只有架子”
    var glassMeshes = [];
    var glassMat = new THREE.MeshPhysicalMaterial({
      map: glassReflTex, color: 0xcbe9f5, transparent: true, opacity: 0.30,
      roughness: 0.14, metalness: 0.12, roughnessMap: microSurf.rough,
      clearcoat: 1.0, clearcoatRoughness: 0.08, reflectivity: 0.96,
      envMapIntensity: 2.2, side: THREE.DoubleSide, depthWrite: false
    });
    var frameMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8ee,
      map: alumFramePBR.map, roughnessMap: alumFramePBR.roughnessMap,
      normalMap: alumFramePBR.normalMap, normalScale: new THREE.Vector2(0.42, 0.42),
      aoMap: alumFramePBR.aoMap, aoMapIntensity: 0.45,
      roughness: 0.32, metalness: 0.48, envMapIntensity: 1.15
    });
    var soilBedMat = new THREE.MeshStandardMaterial({
      map: absTrayTex, bumpMap: absTrayTex, bumpScale: 0.011,
      normalMap: microSurf.normal, normalScale: new THREE.Vector2(0.5, 0.5),
      roughnessMap: microSurf.rough, aoMap: darkSteelPBR.aoMap, aoMapIntensity: 0.5,
      color: 0x30363c, roughness: 0.76, metalness: 0.06, envMapIntensity: 0.18
    });
    var potMat = new THREE.MeshStandardMaterial({
      map: potPlasticPBR.map, roughnessMap: potPlasticPBR.roughnessMap,
      normalMap: potPlasticPBR.normalMap, normalScale: new THREE.Vector2(0.65, 0.65),
      aoMap: potPlasticPBR.aoMap, aoMapIntensity: 0.65,
      roughness: 0.76, metalness: 0.04, envMapIntensity: 0.18
    });
    // 盆内营养土按水分梯度呈现真实干湿明度差异：W1 湿润深黑褐 -> W2 半干褐 -> W3 干燥浅灰褐
    var potSoilMats = {
      W1: new THREE.MeshStandardMaterial({
        map: soilTexSet.map, normalMap: soilTexSet.normal, normalScale: new THREE.Vector2(1.3, 1.3),
        roughnessMap: soilTexSet.rough, color: 0x8a7e74, roughness: 0.86, metalness: 0.0, envMapIntensity: 0.08
      }),
      W2: new THREE.MeshStandardMaterial({
        map: soilTexSet.map, normalMap: soilTexSet.normal, normalScale: new THREE.Vector2(1.3, 1.3),
        roughnessMap: soilTexSet.rough, color: 0xb8a696, roughness: 0.94, metalness: 0.0, envMapIntensity: 0.05
      }),
      W3: new THREE.MeshStandardMaterial({
        map: soilTexSet.map, normalMap: soilTexSet.normal, normalScale: new THREE.Vector2(1.3, 1.3),
        roughnessMap: soilTexSet.rough, color: 0xdccbb8, roughness: 0.98, metalness: 0.0, envMapIntensity: 0.04
      })
    };
    var rootCrownMat = new THREE.MeshStandardMaterial({
      map: potPlasticPBR.map, roughnessMap: microSurf.rough,
      normalMap: microSurf.normal, normalScale: new THREE.Vector2(0.75, 0.75),
      color: 0x8f6b58, roughness: 0.88, metalness: 0.01, envMapIntensity: 0.10
    });
    var contactShadowMat = new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false, opacity: 0.72 });

    /* -------------------- 4. 裁剪式写实试验地块沙盘底座 + 门前水泥小路 + 1.32m 宽巡检通道 -------------------- */
    // 只保留温室周边约 2.4m 宽的真实农田土地（16.8m × 11.8m 精致沙盘地块），四周回归深青色数字孪生科研空间
    var plotW = 16.8, plotD = 11.8;
    var fieldSoilGeo = new THREE.PlaneGeometry(plotW, plotD, 48, 36);
    fieldSoilGeo.rotateX(-Math.PI / 2);
    var fsPos = fieldSoilGeo.attributes.position;
    function fieldHeightAt(fx, fz) {
      var distFromHouse = Math.max(0, Math.max(Math.abs(fx) - 6.3, Math.abs(fz) - 4.3));
      var blend = Math.min(1, distFromHouse / 1.8);
      var edgeFade = Math.min(1, Math.min((plotW * 0.5 - Math.abs(fx)) / 0.6, (plotD * 0.5 - Math.abs(fz)) / 0.6));
      var u = fbm3D(fx * 0.35, 0, fz * 0.35, 901) * 0.055 * blend * Math.max(0, edgeFade);
      return Math.max(-0.022, u);
    }
    for (var fi = 0; fi < fsPos.count; fi++) {
      var fx = fsPos.getX(fi), fz = fsPos.getZ(fi);
      fsPos.setY(fi, fieldHeightAt(fx, fz));
    }
    fsPos.needsUpdate = true;
    fieldSoilGeo.computeVertexNormals();

    var outerGroundMat = new THREE.MeshStandardMaterial({
      map: fieldSoilSet.map,
      normalMap: fieldSoilSet.normalMap, normalScale: new THREE.Vector2(1.85, 1.85),
      roughnessMap: fieldSoilSet.roughnessMap,
      color: 0x564639, roughness: 0.95, metalness: 0.0, envMapIntensity: 0.06
    });
    var outerGround = new THREE.Mesh(fieldSoilGeo, outerGroundMat);
    outerGround.position.y = -0.028;
    outerGround.receiveShadow = true;
    outerGround.visible = true;
    scene.add(outerGround);

    // 地块地质剖面深褐侧裙边与数字孪生青色边界框（顶面下沉至 -0.08，绝不穿出土壤表层）
    var plotPlinth = new THREE.Mesh(
      createBeveledBox(plotW + 0.06, 0.22, plotD + 0.06, 0.015),
      new THREE.MeshStandardMaterial({ color: 0x221a14, roughness: 0.92, metalness: 0.05 })
    );
    plotPlinth.position.y = -0.19;
    scene.add(plotPlinth);
    var plotRim = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(plotW + 0.08, 0.02, plotD + 0.08)),
      new THREE.LineBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.36 })
    );
    plotRim.position.y = -0.036;
    scene.add(plotRim);

    // 温室门前带混凝土路牙的水泥小路（精确延伸至地块西侧边缘 x=-8.4）
    var entryPath = new THREE.Mesh(createBeveledBox(2.2, 0.036, 1.85, 0.008), curbMat);
    entryPath.position.set(-7.3, -0.018, 0);
    entryPath.receiveShadow = true;
    scene.add(entryPath);
    [-0.98, 0.98].forEach(function (kz) {
      var kerb = new THREE.Mesh(createBeveledBox(2.2, 0.065, 0.12, 0.008), darkSteelMat);
      kerb.position.set(-7.3, 0.0, kz);
      scene.add(kerb);
    });
    // 地块北缘低矮试验区围栏（收束在 z = -5.65 边界内）
    var fenceMat = new THREE.MeshStandardMaterial({ color: 0x3b4852, roughness: 0.85, metalness: 0.25 });
    for (var fx = -7.8; fx <= 7.9; fx += 2.6) {
      var fPost = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.72, 8), fenceMat);
      fPost.position.set(fx, 0.33, -5.65);
      scene.add(fPost);
    }
    var fRail = new THREE.Mesh(new THREE.BoxGeometry(15.8, 0.024, 0.024), fenceMat);
    fRail.position.set(0.0, 0.62, -5.65);
    scene.add(fRail);
    var fRailLow = new THREE.Mesh(new THREE.BoxGeometry(15.8, 0.020, 0.020), fenceMat);
    fRailLow.position.set(0.0, 0.30, -5.65);
    scene.add(fRailLow);

    var groundMat = new THREE.MeshStandardMaterial({
      map: fieldSoilSet.map, normalMap: fieldSoilSet.normalMap,
      normalScale: new THREE.Vector2(1.45, 1.45),
      roughnessMap: fieldSoilSet.roughnessMap,
      color: 0x5c4b3d, roughness: 0.94, metalness: 0.0, envMapIntensity: 0.06
    });
    var ground = new THREE.Mesh(createBeveledBox(11.52, 0.04, 7.92, 0.008), groundMat);
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);

    // 农田现实微瑕细节：散落石子 + 沟边低矮野草簇（打破几何完美感）
    (function addFieldImperfections() {
      var rngF = mulberry32(4112);
      var stoneGeo = new THREE.DodecahedronGeometry(0.06, 0);
      var stoneMat = new THREE.MeshStandardMaterial({ color: 0x6f7478, roughness: 0.92, metalness: 0.04 });
      for (var si = 0; si < 22; si++) {
        var st = new THREE.Mesh(stoneGeo, stoneMat);
        var sx = (rngF() > 0.5 ? -1 : 1) * (6.2 + rngF() * 1.8);
        var sz = (rngF() - 0.5) * 10.6;
        if (Math.abs(sz) < 1.15 && sx < 0) continue; // 避开水泥小路
        st.position.set(sx, -0.022 + fieldHeightAt(sx, sz), sz);
        st.scale.set(0.6 + rngF() * 1.1, 0.35 + rngF() * 0.45, 0.6 + rngF() * 1.1);
        st.rotation.set(rngF() * 3, rngF() * 3, rngF() * 3);
        scene.add(st);
      }
      var grassBladeGeo = new THREE.ConeGeometry(0.025, 0.16, 4);
      var wildGrassMat = new THREE.MeshStandardMaterial({ color: 0x3b5828, roughness: 0.85 });
      for (var gi = 0; gi < 18; gi++) {
        var tuft = new THREE.Group();
        var gx = (rngF() > 0.5 ? -1 : 1) * (6.1 + rngF() * 1.7);
        var gz = (rngF() - 0.5) * 10.4;
        if (Math.abs(gz) < 1.2 && gx < 0) continue;
        tuft.position.set(gx, -0.028 + fieldHeightAt(gx, gz), gz);
        for (var bi = 0; bi < 4; bi++) {
          var gb = new THREE.Mesh(grassBladeGeo, wildGrassMat);
          gb.position.set((rngF() - 0.5) * 0.06, 0.06, (rngF() - 0.5) * 0.06);
          gb.rotation.set((rngF() - 0.5) * 0.5, rngF() * 3, (rngF() - 0.5) * 0.5);
          tuft.add(gb);
        }
        scene.add(tuft);
      }
    })();

    // 异步加载本地 Poly Haven CC0 1K 真实摄影 PBR 贴图（concrete / soil / metal），加载成功后自动热升级材质通道
    (function loadLocalCC0Textures() {
      if (typeof THREE.TextureLoader !== 'function') return;
      var tl = new THREE.TextureLoader();
      function applyPBRSet(prefix, rx, ry, targets) {
        function loadTex(suffix, isSRGB, cb) {
          tl.load('assets/pbr/' + prefix + '_' + suffix + '.jpg', function (t) {
            t.wrapS = t.wrapT = THREE.RepeatWrapping;
            t.repeat.set(rx, ry);
            t.anisotropy = 8;
            if (isSRGB) t.encoding = THREE.sRGBEncoding;
            cb(t);
          }, undefined, function () {});
        }
        loadTex('diff', true, function (t) {
          targets.forEach(function (m) { m.map = t; m.needsUpdate = true; });
        });
        loadTex('nor', false, function (t) {
          targets.forEach(function (m) { m.normalMap = t; m.needsUpdate = true; });
        });
        loadTex('rough', false, function (t) {
          targets.forEach(function (m) { m.roughnessMap = t; m.needsUpdate = true; });
        });
      }
      applyPBRSet('soil', 4, 3, [outerGroundMat, groundMat]);
      applyPBRSet('concrete', 3, 2, [curbMat]);
      applyPBRSet('metal', 3, 2, [archMat, frameMat]);
    })();

    var stageGrid = new THREE.GridHelper(16, 16, 0x1a5b66, 0x0d2b38);
    stageGrid.position.y = -0.028;
    stageGrid.visible = false; // 关闭生硬线框网格，让真实土地纹理自然呈现
    scene.add(stageGrid);

    // 室内两条 1.32m 宽纵向科研人员防滑巡检通道（位于 W1-W2 与 W2-W3 之间）+ 前端横向主作业通道
    var aisleMat = new THREE.MeshStandardMaterial({
      map: absTrayTex, roughnessMap: microSurf.rough,
      color: 0x454e56, roughness: 0.88, metalness: 0.05, envMapIntensity: 0.12
    });
    var safetyBorderMat = new THREE.MeshBasicMaterial({ color: 0xd99b26, transparent: true, opacity: 0.65 });
    [-1.18, 1.18].forEach(function (az) {
      var aisleWalk = new THREE.Mesh(createBeveledBox(9.8, 0.014, 0.72, 0.004), aisleMat);
      aisleWalk.position.set(0.2, 0.012, az);
      aisleWalk.receiveShadow = true;
      scene.add(aisleWalk);
      [-0.35, 0.35].forEach(function (sz) {
        var sLine = new THREE.Mesh(new THREE.PlaneGeometry(9.7, 0.025), safetyBorderMat);
        sLine.rotation.x = -Math.PI / 2;
        sLine.position.set(0.2, 0.020, az + sz);
        scene.add(sLine);
      });
    });

    // 前端横向主巡检过道与中央排水明沟格栅
    var walkway = new THREE.Mesh(createBeveledBox(1.12, 0.014, 7.6, 0.004), curbMat);
    walkway.position.set(-5.05, 0.012, 0);
    walkway.receiveShadow = true;
    scene.add(walkway);

    // 温室墙脚紧贴式窄排水沟格栅 + 入口小踏步坡道（取消外围大面积水泥散水，保留真实农田土壤直接衔接墙脚）
    var roadGroup = new THREE.Group();
    roadGroup.visible = false; // 暗夜科研模式默认隐藏外围排水沟格栅与门前坡道

    // 紧贴温室南北矮墙外缘的窄排水沟与不锈钢格栅带
    var grateMat = new THREE.MeshStandardMaterial({
      map: absTrayTex, roughnessMap: galvSteelPBR.roughnessMap,
      color: 0x7e8a92, roughness: 0.45, metalness: 0.65
    });
    [-4.12, 4.12].forEach(function (gz) {
      var gutterRim = new THREE.Mesh(createBeveledBox(11.56, 0.032, 0.30, 0.006), curbMat);
      gutterRim.position.set(0, -0.022, gz);
      gutterRim.receiveShadow = true;
      roadGroup.add(gutterRim);
      var gr = new THREE.Mesh(new THREE.PlaneGeometry(11.48, 0.22), grateMat);
      gr.rotation.x = -Math.PI / 2;
      gr.position.set(0, -0.004, gz);
      roadGroup.add(gr);
    });
    var ramp = new THREE.Mesh(createBeveledBox(0.95, 0.05, 1.95, 0.010), curbMat);
    ramp.position.set(-6.18, 0.005, 0);
    ramp.receiveShadow = true;
    roadGroup.add(ramp);
    scene.add(roadGroup);

    /* -------------------- 5. Venlo 潮汐/滴灌双模苗床 + 三色分床供水总管/电磁阀/压力表 + 双气象站 -------------------- */
    var legGeo = new THREE.CylinderGeometry(0.032, 0.032, 0.56, 12);
    var wheelGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.03, 14);
    wheelGeo.rotateX(Math.PI / 2);
    var waterSheets = {};
    var bedPipes = {};

    // 苗床宽度收缩至 1.04m（留出充足的 1.32m 科研人员巡检过道与水肥管网空间）
    var benchFrameGeo = createBeveledBox(10.24, 0.11, 1.04, 0.015);
    var subTrayGeo = createBeveledBox(10.02, 0.045, 0.92, 0.008);
    var brassMat = new THREE.MeshStandardMaterial({
      map: galvSteelPBR.map, roughnessMap: galvSteelPBR.roughnessMap,
      color: 0xd49b35, roughness: 0.34, metalness: 0.78
    });

    // 从水肥首部 (-5.15, -3.2) 通往三张苗床前端的横向分路供水干管（置于地面管槽，清晰展示水源路径）
    var headerManifold = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 5.9, 14), pvcMat);
    headerManifold.rotation.x = Math.PI / 2;
    headerManifold.position.set(-4.88, 0.16, -0.4);
    scene.add(headerManifold);

    ['W1', 'W2', 'W3'].forEach(function (bed) {
      var bz = BED_Z[bed];
      var pHex = BED_PIPE_HEX[bed];
      var boxMesh = new THREE.Mesh(benchFrameGeo, frameMat);
      boxMesh.position.set(0.15, 0.56, bz);
      boxMesh.castShadow = true; boxMesh.receiveShadow = true;
      scene.add(boxMesh);

      var subMesh = new THREE.Mesh(subTrayGeo, soilBedMat);
      subMesh.position.set(0.15, 0.612, bz);
      subMesh.receiveShadow = true;
      scene.add(subMesh);

      // 支腿 + 滚轮底座 + 接地接触阴影
      [-4.4, 0.15, 4.7].forEach(function (lx) {
        [-0.42, 0.42].forEach(function (lz) {
          var leg = new THREE.Mesh(legGeo, frameMat);
          leg.position.set(lx, 0.28, bz + lz);
          leg.castShadow = true;
          scene.add(leg);
          var wh = new THREE.Mesh(wheelGeo, darkSteelMat);
          wh.position.set(lx, 0.04, bz + lz);
          scene.add(wh);
          var cs = new THREE.Mesh(new THREE.PlaneGeometry(0.24, 0.24), contactShadowMat);
          cs.rotation.x = -Math.PI / 2;
          cs.position.set(lx, 0.012, bz + lz);
          scene.add(cs);
        });
        var brace = new THREE.Mesh(createBeveledBox(0.034, 0.034, 0.84, 0.005), frameMat);
        brace.position.set(lx, 0.22, bz);
        scene.add(brace);
      });

      // 分床色彩编码供水立管 + 黄铜电磁阀 + 立式压力表 + 沿床顶部高亮滴灌主管（凸显水分控制实验核心管网！）
      var lineMat = new THREE.MeshStandardMaterial({
        map: pvcPBR.map, roughnessMap: pvcPBR.roughnessMap,
        color: pHex, roughness: 0.35, metalness: 0.18
      });
      // 床头供水立管（从地面干管升至床面滴灌主管）
      var riserPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.58, 12), lineMat);
      riserPipe.position.set(-4.88, 0.44, bz + 0.36);
      scene.add(riserPipe);

      // 床头工业黄铜电磁阀与状态指示灯环 + 圆形压力表
      var solenoid = new THREE.Mesh(createBeveledBox(0.11, 0.09, 0.09, 0.012), brassMat);
      solenoid.position.set(-4.88, 0.68, bz + 0.36);
      scene.add(solenoid);
      var coil = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.07, 14), lineMat);
      coil.position.set(-4.88, 0.75, bz + 0.36);
      scene.add(coil);
      var gauge = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.034, 0.022, 16), pvcMat);
      gauge.rotation.z = Math.PI / 2;
      gauge.position.set(-4.88, 0.81, bz + 0.36);
      scene.add(gauge);

      // 沿床顶部高亮色彩编码滴灌主支管 + W1 供水流动水滴节点
      var pipeMatBright = new THREE.MeshStandardMaterial({
        map: pvcPBR.map, roughnessMap: pvcPBR.roughnessMap,
        color: pHex, emissive: pHex, emissiveIntensity: bed === 'W1' ? 0.30 : 0.14,
        roughness: 0.35, metalness: 0.18
      });
      var pipeGeo = new THREE.CylinderGeometry(0.024, 0.024, 10.0, 12);
      pipeGeo.rotateZ(Math.PI / 2);
      var pipe = new THREE.Mesh(pipeGeo, pipeMatBright);
      pipe.position.set(0.15, 0.69, bz + 0.36);
      scene.add(pipe);
      bedPipes[bed] = bz + 0.36;
      if (bed === 'W1') {
        global._w1FlowDrops = global._w1FlowDrops || [];
        var dropGeo = new THREE.SphereGeometry(0.030, 10, 10);
        var dropMat = new THREE.MeshBasicMaterial({ color: 0x7dd3fc });
        for (var di = 0; di < 6; di++) {
          var dr = new THREE.Mesh(dropGeo, dropMat);
          dr.position.set(-4.6 + di * 1.6, 0.69, bz + 0.36);
          scene.add(dr);
          global._w1FlowDrops.push({ mesh: dr, phase: di / 6 });
        }
      }

      // 床侧 RS485 土壤水分遥测信号线槽（连接各盆 TDR 土壤湿度探头）
      var busConduit = new THREE.Mesh(createBeveledBox(10.0, 0.022, 0.026, 0.004), darkSteelMat);
      busConduit.position.set(0.15, 0.635, bz - 0.38);
      scene.add(busConduit);

      // 潮汐补水漫灌水膜
      var wMat = new THREE.MeshPhysicalMaterial({
        color: 0x38bdf8, transparent: true, opacity: 0.0, roughness: 0.06, metalness: 0.08,
        clearcoat: 1.0, reflectivity: 0.9, depthWrite: false
      });
      var wMesh = new THREE.Mesh(new THREE.PlaneGeometry(9.94, 0.88), wMat);
      wMesh.rotation.x = -Math.PI / 2;
      wMesh.position.set(0.15, 0.638, bz);
      wMesh.visible = false;
      scene.add(wMesh);
      waterSheets[bed] = { mesh: wMesh, mat: wMat };
    });

    // 真实环境百叶箱传感器立杆（对应 environment.csv Sensor 2 @ W1 / Sensor 1 @ W3，带信号线缆入槽）
    function addEnvSensor(x, z, colHex) {
      var g = new THREE.Group(); g.position.set(x, 0, z);
      var pole = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1.52, 12), frameMat);
      pole.position.y = 0.76; pole.castShadow = true; g.add(pole);
      for (var i = 0; i < 6; i++) {
        var plate = new THREE.Mesh(
          new THREE.CylinderGeometry(0.095, 0.112, 0.02, 18),
          pvcMat
        );
        plate.position.y = 1.32 + i * 0.028; plate.castShadow = true; g.add(plate);
      }
      var ring = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.04, 12), new THREE.MeshBasicMaterial({ color: colHex }));
      ring.position.y = 1.18; g.add(ring);
      scene.add(g);
    }
    addEnvSensor(-4.95, BED_Z.W1 - 0.44, 0x38bdf8);
    addEnvSensor(-4.95, BED_Z.W3 + 0.44, 0xfb7185);

    /* -------------------- 6. Venlo 双连跨玻璃温室建筑（含 W 形桁架腹杆与体积光束） -------------------- */
    var eaveY = 2.85, ridgeY = 3.65, halfSpan = 1.95;
    var roofGroup = new THREE.Group();
    scene.add(roofGroup);

    function addBeam(p1, p2, r, mat, parent, castSh) {
      var dist = p1.distanceTo(p2);
      // 采用 8 边形倒角截面型钢（消除生硬 90° 直角棱边，呈现真实倒角金属高光过渡）
      var bGeo = new THREE.CylinderGeometry(r * 0.56, r * 0.56, dist, 8);
      bGeo.rotateY(Math.PI / 8);
      if (bGeo.attributes.uv && !bGeo.attributes.uv2) bGeo.setAttribute('uv2', bGeo.attributes.uv.clone());
      var mesh = new THREE.Mesh(bGeo, mat || archMat);
      mesh.position.copy(p1).add(p2).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
      if (castSh !== false) mesh.castShadow = true;
      (parent || scene).add(mesh);
    }

    var bayXs = [-5.7, -2.85, 0, 2.85, 5.7];
    var basePlateGeo = createBeveledBox(0.20, 0.028, 0.20, 0.005);
    var anchorBoltGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.036, 6);
    bayXs.forEach(function (ax) {
      [-3.9, 0, 3.9].forEach(function (pz) {
        addBeam(new THREE.Vector3(ax, 0, pz), new THREE.Vector3(ax, eaveY, pz), 0.068);
        var bp = new THREE.Mesh(basePlateGeo, darkSteelMat);
        bp.position.set(ax, 0.015, pz);
        scene.add(bp);
        // 4 颗六角地脚螺栓与柱脚接触阴影
        [-0.065, 0.065].forEach(function (bx) {
          [-0.065, 0.065].forEach(function (bzBolt) {
            var bolt = new THREE.Mesh(anchorBoltGeo, archMat);
            bolt.position.set(ax + bx, 0.032, pz + bzBolt);
            scene.add(bolt);
          });
        });
      });
      // 两跨人字屋顶 + W 形三角形钢构桁架腹杆（次级细檩条关闭投影 castSh=false，彻底消除地面杂乱黑色条纹影子）
      [-1.95, 1.95].forEach(function (cz) {
        var lEave = new THREE.Vector3(ax, eaveY, cz - halfSpan);
        var rEave = new THREE.Vector3(ax, eaveY, cz + halfSpan);
        var ridge = new THREE.Vector3(ax, ridgeY, cz);
        var midBot = new THREE.Vector3(ax, eaveY, cz);
        var lMid = lEave.clone().lerp(ridge, 0.5);
        var rMid = rEave.clone().lerp(ridge, 0.5);
        addBeam(lEave, ridge, 0.048, archMat, roofGroup, true);
        addBeam(rEave, ridge, 0.048, archMat, roofGroup, true);
        addBeam(lEave, rEave, 0.036, archMat, roofGroup, false);
        // W 形腹杆
        addBeam(midBot, ridge, 0.022, archMat, roofGroup, false);
        addBeam(midBot, lMid, 0.020, archMat, roofGroup, false);
        addBeam(midBot, rMid, 0.020, archMat, roofGroup, false);
      });
    });

    [-3.9, 0, 3.9].forEach(function (pz) {
      addBeam(new THREE.Vector3(-5.7, eaveY, pz), new THREE.Vector3(5.7, eaveY, pz), 0.056, archMat, scene, true);
    });
    // 双连跨屋顶中间谷底 (z = 0) 宽厚镀锌排水大天沟 (Center Valley Gutter) + 两端落水管
    var centerGutter = new THREE.Mesh(createBeveledBox(11.56, 0.11, 0.24, 0.014), archMat);
    centerGutter.position.set(0, eaveY + 0.03, 0);
    centerGutter.castShadow = true;
    scene.add(centerGutter);
    [-5.65, 5.65].forEach(function (dx) {
      var cPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, eaveY, 12), pvcMat);
      cPipe.position.set(dx + (dx > 0 ? 0.08 : -0.08), eaveY / 2, 0.10);
      scene.add(cPipe);
    });
    [-1.95, 1.95].forEach(function (cz) {
      addBeam(new THREE.Vector3(-5.7, ridgeY, cz), new THREE.Vector3(5.7, ridgeY, cz), 0.05, archMat, roofGroup, true);
      var slopeLen = Math.hypot(halfSpan, ridgeY - eaveY);
      var ang = Math.atan2(ridgeY - eaveY, halfSpan);
      [-1, 1].forEach(function (side) {
        var pane = new THREE.Mesh(new THREE.PlaneGeometry(11.4, slopeLen), glassMat);
        pane.position.set(0, (eaveY + ridgeY) / 2, cz + side * halfSpan / 2);
        pane.rotation.x = -Math.PI / 2 + side * ang;
        roofGroup.add(pane); glassMeshes.push(pane);
        var midZ = cz + side * halfSpan * 0.5;
        var midY = (eaveY + ridgeY) * 0.5;
        addBeam(new THREE.Vector3(-5.7, midY, midZ), new THREE.Vector3(5.7, midY, midZ), 0.024, archMat, roofGroup, false);
        for (var rx = -4.75; rx <= 4.76; rx += 0.95) {
          addBeam(new THREE.Vector3(rx, eaveY, cz + side * halfSpan), new THREE.Vector3(rx, ridgeY, cz), 0.018, archMat, roofGroup, false);
        }
      });
    });

    // 屋脊电动通风天窗（带与屋脊相连的支撑臂，消除浮空线段穿帮）+ 内部吊装环流风机 + LED 植物补光灯条（带全长供电桥架）
    var innerFans = [];
    [-1.95, 1.95].forEach(function (cz) {
      [-2.85, 2.85].forEach(function (vx) {
        var vent = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 0.75), glassMat);
        vent.position.set(vx, ridgeY + 0.12, cz + 0.34);
        vent.rotation.x = -Math.PI / 2 + 0.38;
        roofGroup.add(vent); glassMeshes.push(vent);
        addBeam(new THREE.Vector3(vx - 1.3, ridgeY + 0.22, cz + 0.65), new THREE.Vector3(vx + 1.3, ridgeY + 0.22, cz + 0.65), 0.020, archMat, roofGroup, false);
        // 天窗两端斜撑臂连接到屋脊梁，彻底消除任何浮空线段
        addBeam(new THREE.Vector3(vx - 1.3, ridgeY, cz), new THREE.Vector3(vx - 1.3, ridgeY + 0.22, cz + 0.65), 0.016, archMat, roofGroup, false);
        addBeam(new THREE.Vector3(vx + 1.3, ridgeY, cz), new THREE.Vector3(vx + 1.3, ridgeY + 0.22, cz + 0.65), 0.016, archMat, roofGroup, false);

        // 环流风机顶部双吊装支架
        addBeam(new THREE.Vector3(vx, 2.66, cz), new THREE.Vector3(vx, eaveY + 0.25, cz), 0.020, darkSteelMat, roofGroup, false);
        var cf = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.22, 20, 1, true), darkSteelMat);
        cf.rotation.z = Math.PI / 2; cf.position.set(vx, 2.66, cz);
        roofGroup.add(cf);
        var fHub = new THREE.Group(); fHub.position.set(vx, 2.66, cz);
        for (var k = 0; k < 3; k++) {
          var bl = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.24, 0.05), archMat);
          bl.rotation.x = (Math.PI * 2 * k) / 3; fHub.add(bl);
        }
        roofGroup.add(fHub);
        innerFans.push(fHub);
      });
    });

    var lampBarGeo = createBeveledBox(1.85, 0.042, 0.14, 0.008);
    var cableTrayGeo = createBeveledBox(11.36, 0.028, 0.045, 0.005); // 桥架全长贯通至前后山墙，绝不悬空
    var hangerRodGeo = new THREE.CylinderGeometry(0.007, 0.007, 0.16, 8);
    var ledGlowMat = new THREE.MeshBasicMaterial({ color: 0xfff5dc });
    ['W1', 'W2', 'W3'].forEach(function (bed) {
      var bz = BED_Z[bed];
      var cTray = new THREE.Mesh(cableTrayGeo, darkSteelMat);
      cTray.position.set(0.0, 2.42, bz);
      scene.add(cTray);
      [-3.1, 0.15, 3.4].forEach(function (lx) {
        var bar = new THREE.Mesh(lampBarGeo, frameMat);
        bar.position.set(lx, 2.28, bz); scene.add(bar);
        var emit = new THREE.Mesh(new THREE.PlaneGeometry(1.76, 0.10), ledGlowMat);
        emit.rotation.x = Math.PI / 2; emit.position.set(lx, 2.258, bz); scene.add(emit);
        [-0.72, 0.72].forEach(function (dx) {
          var rod = new THREE.Mesh(hangerRodGeo, archMat);
          rod.position.set(lx + dx, 2.36, bz); // 吊杆连接灯具与上方桥架
          scene.add(rod);
        });
      });
    });

    // 午后丁达尔空气体积光束 + 漂浮水雾层（仅在日光模式开启）
    var sunShaftGroup = new THREE.Group();
    sunShaftGroup.visible = false;
    var shaftMat = new THREE.MeshBasicMaterial({
      map: shaftTex, transparent: true, opacity: 0.14, blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide, depthWrite: false
    });
    [-3.6, -1.2, 1.2, 3.6].forEach(function (sx) {
      var sh = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 3.4), shaftMat);
      sh.position.set(sx, 1.85, 0.2);
      sh.rotation.z = 0.42;
      sh.rotation.y = 0.35;
      sunShaftGroup.add(sh);
    });
    var mistLayers = [];
    var mistMat = new THREE.MeshBasicMaterial({
      map: mistTex, transparent: true, opacity: 0.24, side: THREE.DoubleSide, depthWrite: false
    });
    [
      [-2.6, 1.15, -1.18, 4.8, 1.6],
      [1.8, 1.22, 1.18, 5.2, 1.7],
      [-0.4, 1.75, 0.0, 6.4, 1.9]
    ].forEach(function (mp, mi) {
      var mPlane = new THREE.Mesh(new THREE.PlaneGeometry(mp[3], mp[4]), mistMat);
      mPlane.position.set(mp[0], mp[1], mp[2]);
      mPlane.rotation.y = (mi - 1) * 0.22;
      sunShaftGroup.add(mPlane);
      mistLayers.push({ mesh: mPlane, baseX: mp[0], baseY: mp[1], phase: mi * 2.1 });
    });
    scene.add(sunShaftGroup);

    // 两侧混凝土矮墙基座（带 3 段圆角倒角 Bevel + 3D FBM 表面微位移）+ 玻璃幕墙 + U型天沟 + 落水管
    var curbBevelGeo = createBeveledBox(11.48, 0.38, 0.16, 0.020, 0.0022, 301);
    var gutterBevelGeo = createBeveledBox(11.54, 0.09, 0.12, 0.012);
    [-3.9, 3.9].forEach(function (wz) {
      var curb = new THREE.Mesh(curbBevelGeo, curbMat);
      curb.position.set(0, 0.19, wz); curb.castShadow = true; curb.receiveShadow = true;
      scene.add(curb);
      var wall = new THREE.Mesh(new THREE.PlaneGeometry(11.4, eaveY - 0.38), glassMat);
      wall.position.set(0, 0.38 + (eaveY - 0.38) / 2, wz); scene.add(wall); glassMeshes.push(wall);
      addBeam(new THREE.Vector3(-5.7, 0.38, wz), new THREE.Vector3(5.7, 0.38, wz), 0.036, archMat, scene, false);
      addBeam(new THREE.Vector3(-5.7, 1.58, wz), new THREE.Vector3(5.7, 1.58, wz), 0.030, archMat, scene, false);
      for (var mx = -5.7; mx <= 5.71; mx += 0.95) {
        addBeam(new THREE.Vector3(mx, 0.38, wz), new THREE.Vector3(mx, eaveY, wz), 0.022, archMat, scene, false);
      }
      var gutter = new THREE.Mesh(gutterBevelGeo, archMat);
      gutter.position.set(0, eaveY + 0.02, wz); scene.add(gutter);
      [-5.65, 5.65].forEach(function (dx) {
        var ds = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, eaveY, 12), pvcMat);
        ds.position.set(dx, eaveY / 2, wz + (wz > 0 ? 0.09 : -0.09)); scene.add(ds);
      });
    });

    // 前后山墙、双开玻璃推拉门、后端带铝合金防雨百叶与接线管路的负压排风风机、前端 7090 蜂窝湿帘墙
    var fanBlades = [];
    var padMat = new THREE.MeshStandardMaterial({
      map: padTex.map, normalMap: padTex.normal, normalScale: new THREE.Vector2(1.1, 1.1),
      roughnessMap: microSurf.rough, roughness: 0.88, metalness: 0.04, envMapIntensity: 0.14
    });
    var gCurbBevelGeo = createBeveledBox(0.16, 0.38, 2.7, 0.018, 0.0022, 309);
    var fanBoxBevelGeo = createBeveledBox(0.22, 0.82, 0.82, 0.018);
    var louverSlatGeo = createBeveledBox(0.024, 0.08, 0.74, 0.004);
    var padFrameBevelGeo = createBeveledBox(0.14, 1.38, 1.98, 0.015);
    [-5.7, 5.7].forEach(function (gx) {
      [-2.55, 2.55].forEach(function (cz) {
        var gCurb = new THREE.Mesh(gCurbBevelGeo, curbMat);
        gCurb.position.set(gx, 0.19, cz); gCurb.castShadow = true; gCurb.receiveShadow = true;
        scene.add(gCurb);
      });
      // 山墙两侧与拱顶封闭玻璃幕墙（补齐前后山墙玻璃皮层）
      [-2.4, 2.4].forEach(function (gzPane) {
        var gWall = new THREE.Mesh(new THREE.PlaneGeometry(2.8, eaveY - 0.38), glassMat);
        gWall.rotation.y = Math.PI / 2;
        gWall.position.set(gx, 0.38 + (eaveY - 0.38) / 2, gzPane);
        scene.add(gWall); glassMeshes.push(gWall);
      });
      [-2.9, -1.9, -0.95, 0.95, 1.9, 2.9].forEach(function (gz) {
        var topY = gz < 0 ? (eaveY + (1.95 - Math.abs(gz + 1.95)) * (ridgeY - eaveY) / halfSpan)
                          : (eaveY + (1.95 - Math.abs(gz - 1.95)) * (ridgeY - eaveY) / halfSpan);
        addBeam(new THREE.Vector3(gx, 0, gz), new THREE.Vector3(gx, topY, gz), 0.028, archMat, scene, false);
      });
      addBeam(new THREE.Vector3(gx, 2.18, -0.95), new THREE.Vector3(gx, 2.18, 0.95), 0.046, archMat, scene, false);
      var doorGlass = new THREE.Mesh(new THREE.PlaneGeometry(1.86, 2.14), glassMat);
      doorGlass.rotation.y = Math.PI / 2; doorGlass.position.set(gx, 1.07, 0); scene.add(doorGlass); glassMeshes.push(doorGlass);
      [-0.08, 0.08].forEach(function (hz) {
        var handle = new THREE.Mesh(createBeveledBox(0.06, 0.28, 0.03, 0.005), darkSteelMat);
        handle.position.set(gx, 1.05, hz); scene.add(handle);
      });
      if (gx > 0) {
        [-1.95, 1.95].forEach(function (fz) {
          // 工业负压风机镀锌外框 + 圆形导流风筒 + 6 片活动百叶窗 + 供电穿线管 + 状态绿灯（彻底告别“纯灰方块”）
          var fBox = new THREE.Mesh(fanBoxBevelGeo, archMat);
          fBox.position.set(gx + 0.06, 2.12, fz); scene.add(fBox);
          var shroud = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.26, 24, 1, true), darkSteelMat);
          shroud.rotation.z = Math.PI / 2;
          shroud.position.set(gx + 0.06, 2.12, fz); scene.add(shroud);
          for (var li = -2; li <= 2; li++) {
            var slat = new THREE.Mesh(louverSlatGeo, frameMat);
            slat.position.set(gx - 0.06, 2.12 + li * 0.12, fz);
            slat.rotation.z = 0.28;
            scene.add(slat);
          }
          // 风机供电金属穿线管与运行指示灯
          addBeam(new THREE.Vector3(gx - 0.02, 0.38, fz + 0.42), new THREE.Vector3(gx - 0.02, 2.12, fz + 0.42), 0.018, darkSteelMat, scene, false);
          var fLed = new THREE.Mesh(new THREE.SphereGeometry(0.018, 8, 8), new THREE.MeshBasicMaterial({ color: 0x3ddc84 }));
          fLed.position.set(gx - 0.06, 2.48, fz + 0.34); scene.add(fLed);

          var hub = new THREE.Group(); hub.position.set(gx + 0.14, 2.12, fz);
          for (var bi = 0; bi < 4; bi++) {
            var bMesh = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.30, 0.09), darkSteelMat);
            bMesh.rotation.x = (Math.PI / 2) * bi; hub.add(bMesh);
          }
          scene.add(hub); fanBlades.push(hub);
        });
      } else {
        [-2.35, 2.35].forEach(function (pz) {
          // 银色铝合金外框 + 7090 棕褐蜂窝湿帘纸 + 顶部白色 PVC 进水横管/弯头 + 底部接水回水槽
          var padFrame = new THREE.Mesh(padFrameBevelGeo, archMat);
          padFrame.position.set(gx - 0.05, 1.08, pz); scene.add(padFrame);
          var padCore = new THREE.Mesh(new THREE.BoxGeometry(0.16, 1.28, 1.88), padMat);
          padCore.position.set(gx - 0.05, 1.08, pz); scene.add(padCore);
          var sprayPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 2.02, 12), pvcMat);
          sprayPipe.rotation.x = Math.PI / 2;
          sprayPipe.position.set(gx - 0.07, 1.81, pz); scene.add(sprayPipe);
          var feedDown = new THREE.Mesh(new THREE.CylinderGeometry(0.020, 0.020, 1.44, 10), pvcMat);
          feedDown.position.set(gx - 0.07, 1.10, pz + (pz > 0 ? 0.98 : -0.98)); scene.add(feedDown);
          var returnGutter = new THREE.Mesh(createBeveledBox(0.22, 0.09, 2.04, 0.010), darkSteelMat);
          returnGutter.position.set(gx - 0.06, 0.39, pz); scene.add(returnGutter);
        });
      }
    });

    // 水肥一体化三桶首部系统（带黑色螺纹桶盖、抱箍环、液位观察条与出水管）+ 工业倒角 PLC 控制柜
    (function addFertigationSystem() {
      var tankBaseMat = {
        roughnessMap: microSurf.rough, normalMap: microSurf.normal,
        normalScale: new THREE.Vector2(0.45, 0.45), roughness: 0.42, metalness: 0.08, envMapIntensity: 0.28
      };
      var specs = [
        { x: -5.32, z: -3.38, r: 0.26, h: 0.92, col: 0x1a68b5 },
        { x: -4.75, z: -3.42, r: 0.19, h: 0.74, col: 0xd49626 },
        { x: -4.26, z: -3.42, r: 0.19, h: 0.74, col: 0x2d8f5e }
      ];
      specs.forEach(function (sp) {
        var tk = new THREE.Mesh(new THREE.CylinderGeometry(sp.r, sp.r, sp.h, 24), new THREE.MeshStandardMaterial(Object.assign({ color: sp.col }, tankBaseMat)));
        tk.position.set(sp.x, sp.h * 0.5, sp.z); tk.castShadow = true; scene.add(tk);
        // 黑色螺纹桶盖
        var lid = new THREE.Mesh(new THREE.CylinderGeometry(sp.r * 0.44, sp.r * 0.44, 0.05, 16), darkSteelMat);
        lid.position.set(sp.x, sp.h + 0.02, sp.z); scene.add(lid);
        // 上下两条黑色抱箍环
        [0.25, 0.74].forEach(function (hr) {
          var hoop = new THREE.Mesh(new THREE.CylinderGeometry(sp.r + 0.008, sp.r + 0.008, 0.032, 24, 1, true), darkSteelMat);
          hoop.position.set(sp.x, sp.h * hr, sp.z); scene.add(hoop);
        });
        // 桶顶出水管弯向横向供水干管 (-4.88)
        var outPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.48, 8), pvcMat);
        outPipe.position.set(sp.x, sp.h * 0.65, sp.z + 0.24);
        outPipe.rotation.x = Math.PI / 2; scene.add(outPipe);
      });
      var cab = new THREE.Mesh(createBeveledBox(0.32, 0.92, 0.44, 0.016), frameMat);
      cab.position.set(-5.32, 0.76, -2.72); cab.castShadow = true; scene.add(cab);
      [0x3ddc84, 0x38bdf8, 0xffb020].forEach(function (lc, li) {
        var led = new THREE.Mesh(new THREE.SphereGeometry(0.018, 10, 10), new THREE.MeshBasicMaterial({ color: lc }));
        led.position.set(-5.15, 1.08, -2.62 - li * 0.08); scene.add(led);
      });
    })();

    // 龙门纵向行车导轨
    var railBevelGeo = createBeveledBox(11.4, 0.055, 0.055, 0.008);
    [-3.10, 3.10].forEach(function (rz) {
      var rail = new THREE.Mesh(railBevelGeo, archMat);
      rail.position.set(0, 2.55, rz); scene.add(rail);
    });

    /* -------------------- 7. 33 观测株 + 30 伴生株 生理梯度形态甜菜 + TDR 土壤水分探头 + 分色滴灌软管 -------------------- */
    var leafGeoVariants = [
      createParametricBeetLeafGeometry(1),
      createParametricBeetLeafGeometry(2),
      createParametricBeetLeafGeometry(3),
      createParametricBeetLeafGeometry(4)
    ];
    var potGeo = createDeformedPotGeometry(611);
    var potRimGeo = createBeveledPotRimGeometry(619);
    var potSoilGeo = createDisplacedSoilGeometry(0.148, 0.024, 505);
    var rootCrownGeo = createSugarBeetRootCrownGeometry(808);
    var ringGeo = new THREE.RingGeometry(0.185, 0.235, 28);
    ringGeo.rotateX(-Math.PI / 2);
    var potShadowGeo = new THREE.PlaneGeometry(0.40, 0.40);
    potShadowGeo.rotateX(-Math.PI / 2);

    // TDR 土壤水分/温度一体式传感器探头几何与材质
    var tdrCapGeo = createBeveledBox(0.036, 0.046, 0.024, 0.005);
    var tdrProngGeo = new THREE.CylinderGeometry(0.0025, 0.0025, 0.06, 8);
    var tdrHeadMat = new THREE.MeshStandardMaterial({ color: 0x0ea5e9, roughness: 0.32, metalness: 0.15 });
    var emitterStakeGeo = new THREE.ConeGeometry(0.012, 0.055, 10);
    emitterStakeGeo.rotateX(Math.PI);

    var plantMeshes = [];
    var pickableObjects = [];
    var selectedUnit = null;
    var hoverUnit = null;

    // 找出 W2 与 W3 中残差温升最高的代表性热点株位（用于默认展示带引线的完整科学标注，彻底替代无意义的孤立 "L3"）
    var defaultCalloutUnits = {};
    ['W2', 'W3'].forEach(function (b) {
      var bList = U.filter(function (u) { return u.bed === b && !u.isCtrl; }).slice().sort(function (a, c) { return c.resid - a.resid; });
      if (bList.length > 0) defaultCalloutUnits[bList[0].unit] = true;
    });

    /**
     * 构建具有真实水分胁迫生理形态梯度的甜菜冠层：
     * - W1 (100% FC): 株高挺拔 (heightScale 1.08~1.16)，叶柄直立，全叶饱满翠绿；
     * - W2 (65% FC):  株高略降 (heightScale 0.86~0.95)，外层老叶适度萎蔫下垂且叶缘向心失绿，中内层叶保持橄榄绿；
     * - W3 (35% FC):  株高显著塌陷 (heightScale 0.62~0.75)，外层叶贴盆边重度翻卷萎蔫、叶缘金黄焦褐，心叶矮小余绿。
     */
    function buildBeetCanopy(parent, outerMat, midMat, coreMat, u, seed, unitId, pickable) {
      var rng = mulberry32(seed);
      var crown = new THREE.Mesh(rootCrownGeo, rootCrownMat);
      crown.position.y = 0.222;
      crown.rotation.y = rng() * Math.PI * 2;
      crown.castShadow = true; crown.receiveShadow = true;
      parent.add(crown);

      var stressSeverity = u.isCtrl ? 0.0 : Math.min(1.0, Math.max(0.15, (u.resid / 5.4) * (u.bed === 'W3' ? 1.05 : 0.72)));
      var bioJitter = (rng() - 0.5) * 0.22; // 加大个体株高、冠幅随机差异，彻底打破“阅兵方阵”复制粘贴感
      var heightScale = u.isCtrl
        ? (1.06 + rng() * 0.16)
        : (u.bed === 'W2' ? Math.max(0.68, 0.94 - stressSeverity * 0.16 + bioJitter)
                          : Math.max(0.50, 0.74 - stressSeverity * 0.20 + bioJitter));
      var wiltTilt = u.isCtrl
        ? 0.0
        : (u.bed === 'W2' ? (0.16 + stressSeverity * 0.22 + rng() * 0.08) : (0.32 + stressSeverity * 0.32 + rng() * 0.10));
      var curlNarrow = u.isCtrl ? 1.0 : (1.0 - stressSeverity * 0.25);

      var layers = [
        { n: 5, tilt: Math.min(1.42, 0.78 + wiltTilt * 1.15), scale: 1.34, hMul: heightScale * 0.92, wMul: curlNarrow, offset: rng() * 2.4, mat: outerMat },
        { n: 5, tilt: Math.min(1.22, 0.48 + wiltTilt * 0.85), scale: 1.06, hMul: heightScale * 1.00, wMul: 0.95 + 0.05 * curlNarrow, offset: rng() * 2.4 + 0.6, mat: midMat },
        { n: 4, tilt: Math.min(0.78, 0.20 + wiltTilt * 0.40), scale: 0.70, hMul: heightScale * 1.05, wMul: 1.0, offset: rng() * 2.4 + 1.2, mat: coreMat }
      ];
      layers.forEach(function (lay, lIdx) {
        for (var li = 0; li < lay.n; li++) {
          // 约 12% 概率自然脱落个别外层老叶，形成真实生物株型差异
          if (lIdx === 0 && rng() < 0.12) continue;
          var vGeo = leafGeoVariants[(li + lIdx + Math.floor(rng() * 4)) % leafGeoVariants.length];
          var leaf = new THREE.Mesh(vGeo, lay.mat);
          var sc = lay.scale * (0.82 + rng() * 0.34);
          leaf.position.y = 0.224;
          leaf.rotation.order = 'YXZ';
          leaf.rotation.y = (Math.PI * 2 * li) / lay.n + lay.offset + (rng() - 0.5) * 0.28;
          leaf.rotation.x = lay.tilt + (rng() - 0.5) * 0.15;
          leaf.rotation.z = (rng() - 0.5) * 0.14;
          leaf.scale.set(
            sc * lay.wMul * (0.92 + rng() * 0.16),
            sc * lay.hMul,
            sc * (0.92 + rng() * 0.16)
          );
          leaf.castShadow = true;
          leaf.receiveShadow = true;
          if (pickable) {
            leaf.userData.unit = unitId;
            pickableObjects.push(leaf);
          }
          parent.add(leaf);
        }
      });
    }

    var bedTubeMats = {
      W1: new THREE.MeshStandardMaterial({ color: 0x0ea5e9, roughness: 0.35 }),
      W2: new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.40 }),
      W3: new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.42 })
    };

    ['W1', 'W2', 'W3'].forEach(function (bed, bi) {
      var bedUnits = U.filter(function (u) { return u.bed === bed; });
      allCodes.forEach(function (code, ci) {
        var px = -4.25 + (allCodes.length > 1 ? ci * (8.8 / (allCodes.length - 1)) : 0);
        var pz = BED_Z[bed];
        var u = bedUnits.filter(function (item) { return item.code === code; })[0];
        if (!u && bed === 'W1') {
          u = { unit: 'W1_' + code, bed: 'W1', code: code, grade: 0, resid: 0, z: 0, isCtrl: true };
        }
        if (!u) return;

        var pRng = mulberry32((bi + 1) * 331 + ci * 97);
        var jx = (pRng() - 0.5) * 0.042;
        var jz = (pRng() - 0.5) * 0.036;
        var group = new THREE.Group();
        group.position.set(px + jx, 0.635, pz + jz);
        group.rotation.y = (pRng() - 0.5) * 0.35;
        group.rotation.z = (pRng() - 0.5) * 0.025;

        // 盆底接触阴影
        var ps = new THREE.Mesh(potShadowGeo, contactShadowMat);
        ps.position.y = 0.004;
        group.add(ps);

        // 育苗盆 + 盆沿 + 3D 凹凸营养土
        var pot = new THREE.Mesh(potGeo, potMat);
        pot.position.y = 0.11; pot.castShadow = true; pot.receiveShadow = true;
        pot.userData.unit = u.unit; group.add(pot); pickableObjects.push(pot);

        var potRim = new THREE.Mesh(potRimGeo, potMat);
        potRim.position.y = 0.215; group.add(potRim);

        var pSoil = new THREE.Mesh(potSoilGeo, potSoilMats[bed] || potSoilMats.W1);
        pSoil.position.y = 0.22; pSoil.receiveShadow = true; group.add(pSoil);

        // 真实 3D 分色滴灌毛管 + 盆内斜插滴箭（清晰展示每一盆的水分供给通路）
        var pipeLocalZ = bedPipes[bed] - (pz + jz);
        var hoseCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(0.02, 0.045, pipeLocalZ),
          new THREE.Vector3(0.04, 0.20, pipeLocalZ * 0.62),
          new THREE.Vector3(0.06, 0.285, 0.11),
          new THREE.Vector3(0.06, 0.23, 0.065)
        ]);
        var tMat = bedTubeMats[bed] || bedTubeMats.W1;
        var hose = new THREE.Mesh(new THREE.TubeGeometry(hoseCurve, 10, 0.0068, 6, false), tMat);
        group.add(hose);
        var stake = new THREE.Mesh(emitterStakeGeo, tMat);
        stake.position.set(0.06, 0.235, 0.065);
        group.add(stake);

        // 盆内 TDR 土壤体积含水率传感器探头 + 不锈钢探针 + 连接至床侧 RS485 总线的信号线
        var tdrCap = new THREE.Mesh(tdrCapGeo, tdrHeadMat);
        tdrCap.position.set(-0.075, 0.255, -0.065);
        group.add(tdrCap);
        [-0.082, -0.068].forEach(function (pxProng) {
          var prong = new THREE.Mesh(tdrProngGeo, archMat);
          prong.position.set(pxProng, 0.225, -0.065);
          group.add(prong);
        });
        var sigCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(-0.075, 0.26, -0.075),
          new THREE.Vector3(-0.09, 0.16, -0.22),
          new THREE.Vector3(-0.08, 0.005, -0.36)
        ]);
        var sigWire = new THREE.Mesh(new THREE.TubeGeometry(sigCurve, 8, 0.0038, 5, false), darkSteelMat);
        group.add(sigWire);

        // 底座状态环 + 残差模式呼吸热辐射光晕
        var colHex = gradeHex(u.grade, u.isCtrl);
        var ringMat = new THREE.MeshBasicMaterial({ color: colHex, transparent: true, opacity: 0.72, side: THREE.DoubleSide });
        var ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.y = 0.008; group.add(ring);

        var pulseGeo = new THREE.RingGeometry(0.19, 0.35, 24);
        pulseGeo.rotateX(-Math.PI / 2);
        var pulseMat = new THREE.MeshBasicMaterial({ color: residHex(u.resid), transparent: true, opacity: 0.45, side: THREE.DoubleSide });
        var pulseMesh = new THREE.Mesh(pulseGeo, pulseMat);
        pulseMesh.position.y = 0.012; pulseMesh.visible = false;
        group.add(pulseMesh);

        // 三层生理差异化甜菜叶片材质：缺水植株外叶采用向心叶缘失绿纹理 (mapChlorosis)，内层嫩叶保留鲜活叶绿素纹理 (map)
        var useChlOuter = !u.isCtrl && (u.grade >= 1 || u.resid > 1.8);
        var useChlMid = !u.isCtrl && (u.bed === 'W3' && (u.grade >= 2 || u.resid > 3.0));
        var leafMat = new THREE.MeshStandardMaterial({
          map: useChlOuter ? leafTexSet.mapChlorosis : leafTexSet.map,
          alphaMap: leafTexSet.alpha,
          alphaTest: 0.45,
          normalMap: leafTexSet.normal,
          normalScale: new THREE.Vector2(0.85, 0.85),
          roughnessMap: leafTexSet.rough,
          aoMap: leafTexSet.ao,
          aoMapIntensity: 0.72,
          color: leafColorHex(u, mode),
          emissive: u.isCtrl ? 0x0a2910 : 0x0c1e08,
          emissiveIntensity: 0.14,
          roughness: u.isCtrl ? 0.44 : 0.56,
          metalness: 0.02,
          envMapIntensity: 0.52,
          side: THREE.DoubleSide
        });
        var midMat = leafMat.clone();
        midMat.map = useChlMid ? leafTexSet.mapChlorosis : leafTexSet.map;
        midMat.color.setHex(leafColorHex(u, mode)).offsetHSL(0.015, 0.04, 0.03);

        var coreMat = leafMat.clone();
        coreMat.map = leafTexSet.map; // 中心新叶始终保持叶绿素纹理
        coreMat.color.setHex(u.isCtrl ? 0x2a8038 : (u.bed === 'W2' ? 0x3b8032 : 0x4c7e2c));

        buildBeetCanopy(group, leafMat, midMat, coreMat, u, (bi + 1) * 1000 + ci * 37, u.unit, true);

        scene.add(group);
        plantMeshes.push({
          unit: u.unit, data: u, group: group, leafMat: leafMat, midMat: midMat, coreMat: coreMat,
          ring: ring, ringMat: ringMat, pulseMesh: pulseMesh, pulseMat: pulseMat
        });

        // 伴生甜菜盆栽（同样遵循 W1/W2/W3 株高、萎蔫度与向心失绿生理梯度）
        if (ci < allCodes.length - 1) {
          var stepX = 8.8 / (allCodes.length - 1);
          var fg = new THREE.Group();
          fg.position.set(px + stepX * 0.5 + (pRng() - 0.5) * 0.045, 0.635, pz + (pRng() - 0.5) * 0.035);
          fg.rotation.y = (pRng() - 0.5) * 0.5;
          var fps = new THREE.Mesh(potShadowGeo, contactShadowMat); fps.position.y = 0.004; fg.add(fps);
          var fPot = new THREE.Mesh(potGeo, potMat); fPot.position.y = 0.11; fPot.castShadow = true; fg.add(fPot);
          var fRim = new THREE.Mesh(potRimGeo, potMat); fRim.position.y = 0.215; fg.add(fRim);
          var fSoil = new THREE.Mesh(potSoilGeo, potSoilMats[bed] || potSoilMats.W1); fSoil.position.y = 0.22; fg.add(fSoil);
          var fOuter = leafMat.clone();
          var fMid = midMat.clone();
          var fCore = coreMat.clone();
          buildBeetCanopy(fg, fOuter, fMid, fCore, u, (bi + 1) * 5000 + ci * 71, '', false);
          scene.add(fg);
        }
      });
    });

    /* -------------------- 8. 工业级三轴表型龙门机器人（全倒角构件 + PBR 表面微瑕） -------------------- */
    var gantryGroup = new THREE.Group();
    var bridge = new THREE.Mesh(createBeveledBox(0.09, 0.08, 6.28, 0.010), archMat);
    bridge.position.y = 2.55; bridge.castShadow = true; gantryGroup.add(bridge);
    var chain = new THREE.Mesh(createBeveledBox(0.04, 0.04, 3.2, 0.006), darkSteelMat);
    chain.position.set(0.06, 2.61, -1.4); gantryGroup.add(chain);

    var truckBevelGeo = createBeveledBox(0.32, 0.12, 0.16, 0.012);
    [-3.10, 3.10].forEach(function (tz) {
      var truck = new THREE.Mesh(truckBevelGeo, darkSteelMat);
      truck.position.set(0, 2.55, tz); gantryGroup.add(truck);
      var motor = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.16, 16), darkSteelMat);
      motor.rotation.x = Math.PI / 2; motor.position.set(0.14, 2.62, tz); gantryGroup.add(motor);
    });

    // 多模态四镜头工业防护云台（Thermal 紫金锗镜片 + RGB + HSI + Depth）
    var camPod = new THREE.Group();
    var camBox = new THREE.Mesh(createBeveledBox(0.28, 0.17, 0.48, 0.014), frameMat);
    camBox.position.y = 2.38; camBox.castShadow = true; camPod.add(camBox);
    [
      { z: -0.15, r: 0.044, col: 0xff8c42, glass: 0x6b3f99 },
      { z: -0.04, r: 0.036, col: 0x4a9eff, glass: 0x1a4d80 },
      { z: 0.07,  r: 0.040, col: 0x35e0c0, glass: 0x145a52 },
      { z: 0.17,  r: 0.028, col: 0x94a3b8, glass: 0x222830 }
    ].forEach(function (ln) {
      var lBody = new THREE.Mesh(new THREE.CylinderGeometry(ln.r, ln.r, 0.07, 16), darkSteelMat);
      lBody.position.set(0, 2.26, ln.z); camPod.add(lBody);
      var lGlass = new THREE.Mesh(new THREE.CircleGeometry(ln.r * 0.72, 16), new THREE.MeshPhysicalMaterial({ color: ln.glass, roughness: 0.05, metalness: 0.6, clearcoat: 1.0 }));
      lGlass.rotation.x = Math.PI / 2; lGlass.position.set(0, 2.223, ln.z); camPod.add(lGlass);
      var lRing = new THREE.Mesh(new THREE.RingGeometry(ln.r * 0.72, ln.r * 0.96, 16), new THREE.MeshBasicMaterial({ color: ln.col, side: THREE.DoubleSide }));
      lRing.rotation.x = Math.PI / 2; lRing.position.set(0, 2.222, ln.z); camPod.add(lRing);
    });
    gantryGroup.add(camPod);

    var scanCone = new THREE.Mesh(
      new THREE.ConeGeometry(2.4, 1.55, 24, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false })
    );
    scanCone.position.y = 1.44; gantryGroup.add(scanCone);

    var laserLine = new THREE.Mesh(
      new THREE.BoxGeometry(0.03, 0.012, 5.8),
      new THREE.MeshBasicMaterial({ color: 0x35e0c0, transparent: true, opacity: 0.65 })
    );
    laserLine.position.y = 0.88; gantryGroup.add(laserLine);
    scene.add(gantryGroup);

    /* -------------------- 9. 零重叠屏幕防碰撞矢量引线标牌 (Viewport Clamping + Collision Avoidance) -------------------- */
    var labelPool = [];
    // 将三张床的锚点沿 X 轴错开：W3 置左前端 (-1.2)，彻底避开右后侧高危株位 W3_I8 (+8.53℃) 的垂直引线
    var BED_ANCHOR_X = { W1: -3.4, W2: -0.6, W3: -1.3 };
    var BED_ANCHOR_Y = { W1: 1.16, W2: 1.44, W3: 1.14 };
    ['W1', 'W2', 'W3'].forEach(function (bed) {
      var col = BED_COL[bed];
      var wrapDiv = document.createElement('div');
      wrapDiv.style.cssText = 'position:absolute;transform:translate(-50%,-100%);pointer-events:none;z-index:4;';
      wrapDiv.innerHTML =
        '<div style="padding:3px 9px;border-radius:6px;background:rgba(4,16,26,0.88);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);' +
          'border:1px solid rgba(255,255,255,0.22);border-left:3.5px solid ' + col + ';box-shadow:0 4px 12px rgba(0,0,0,0.48);white-space:nowrap;">' +
          '<div style="font-size:11px;font-weight:700;color:#ffffff;letter-spacing:0.2px;"><span style="color:' + col + ';margin-right:4px;">●</span>' + BED_N[bed] + '</div>' +
        '</div>' +
        '<div class="ldr-line" style="width:1.5px;height:18px;background:linear-gradient(180deg,' + col + ',rgba(255,255,255,0.3));margin:0 auto;"></div>' +
        '<div style="width:6px;height:6px;border-radius:50%;background:' + col + ';box-shadow:0 0 6px ' + col + ';margin:-1px auto 0;"></div>';
      labelsEl.appendChild(wrapDiv);
      labelPool.push({ el: wrapDiv, pos: new THREE.Vector3(BED_ANCHOR_X[bed], BED_ANCHOR_Y[bed], BED_Z[bed]), estW: 156, estH: 48 });
    });

    plantMeshes.forEach(function (pm) {
      if (pm.data.isCtrl) return;
      var div = document.createElement('div');
      div.style.cssText = 'position:absolute;transform:translate(-50%,-100%);pointer-events:none;z-index:5;white-space:nowrap;';
      labelsEl.appendChild(div);
      pm.labelEl = div;
    });

    function effectiveUnitResid(u) {
      if (global._replayDayIdx >= 0 && Array.isArray(u.daily) && global._replayDayIdx < u.daily.length) {
        return u.daily[global._replayDayIdx];
      }
      var rScale = (global._replayScale !== undefined) ? global._replayScale : 1.0;
      return u.resid * rScale;
    }
    function dynamicGrade(u, effResid) {
      if (u.isCtrl) return 0;
      if (global._replayDayIdx >= 0 && Array.isArray(u.daily)) {
        var mu = (D.summary && typeof D.summary.pre_mean === 'number') ? D.summary.pre_mean : 0.289;
        var sd = (D.summary && typeof D.summary.baseline_sd === 'number') ? D.summary.baseline_sd : 1.753;
        var z = (effResid - mu) / sd;
        var cuts = (D.summary && D.summary.z_cuts) ? D.summary.z_cuts : [1.645, 3.0, 4.5];
        var g = 0;
        for (var i = 0; i < cuts.length; i++) if (z >= cuts[i]) g++;
        return g;
      }
      return u.grade;
    }
    function updatePlantVisuals() {
      plantMeshes.forEach(function (pm) {
        var u = pm.data;
        var effResid = effectiveUnitResid(u);
        var dynG = dynamicGrade(u, effResid);
        var cHex;
        if (mode === 'resid') {
          cHex = u.isCtrl ? 0x1e6091 : residHex(effResid);
        } else if (u.isCtrl) {
          cHex = 0x1c6326;
        } else if (u.bed === 'W2') {
          cHex = [0x24682a, 0x4b7726, 0x6c7c24, 0x867822][dynG] || 0x4b7726;
        } else {
          cHex = [0x2c6828, 0x5c7424, 0x827422, 0x986e24][dynG] || 0x827422;
        }
        pm.leafMat.color.setHex(cHex);
        if (pm.midMat) pm.midMat.color.setHex(cHex).offsetHSL(0.015, 0.04, 0.03);
        pm.coreMat.color.setHex(mode === 'resid' ? cHex : (u.isCtrl ? 0x2a8038 : (u.bed === 'W2' ? 0x3b8032 : 0x4c7e2c)));
        pm.pulseMesh.visible = (mode === 'resid' && !u.isCtrl);
        pm.pulseMat.color.setHex(residHex(effResid));
        var ringHex = (mode === 'resid') ? (u.isCtrl ? 0x38bdf8 : residHex(effResid)) : gradeHex(dynG, u.isCtrl);
        pm.ringMat.color.setHex(selectedUnit === u.unit ? 0xffffff : ringHex);
        pm.ringMat.opacity = (selectedUnit === u.unit || hoverUnit === u.unit) ? 1.0 : 0.65;
      });
    }
    global._onReplayTick = updatePlantVisuals;

    // 屏幕空间矩形碰撞避让 + 视口边界钳制算法：彻底杜绝叠压、遮挡与侧视飞出画外
    function updateLabels(W, H) {
      var v = new THREE.Vector3();
      var placedBoxes = [];

      function placeNonOverlapping(el, rawX, rawY, boxW, boxH, minLineH, maxLineH) {
        var padX = boxW * 0.5 + 10;
        var sx = Math.max(padX, Math.min(W - padX, rawX));
        var lineH = minLineH;
        var sy = Math.max(boxH + 8, Math.min(H - 14, rawY));

        // 碰撞检测：如果与已摆放标签重叠，则阶梯上移（拉长引线），仍重叠则微调水平位置
        for (var attempt = 0; attempt < 8; attempt++) {
          var topY = sy - boxH;
          var leftX = sx - boxW * 0.5;
          var rightX = sx + boxW * 0.5;
          var hit = false;
          for (var k = 0; k < placedBoxes.length; k++) {
            var b = placedBoxes[k];
            if (leftX < b.r + 6 && rightX > b.l - 6 && topY < b.b + 5 && sy > b.t - 5) {
              hit = true;
              if (sy - 26 >= boxH + 8) {
                sy -= 26;
                lineH = Math.min(maxLineH, lineH + 26);
              } else {
                sx = Math.min(W - padX, Math.max(padX, sx + (attempt % 2 === 0 ? 96 : -96)));
              }
              break;
            }
          }
          if (!hit) break;
        }
        placedBoxes.push({ l: sx - boxW * 0.5, r: sx + boxW * 0.5, t: sy - boxH, b: sy });
        el.style.left = sx.toFixed(1) + 'px';
        el.style.top = sy.toFixed(1) + 'px';
        var ldr = el.querySelector('.ldr-line');
        if (ldr) ldr.style.height = lineH + 'px';
      }

      labelPool.forEach(function (lb) {
        v.copy(lb.pos).project(camera);
        if (v.z > 1) { lb.el.style.display = 'none'; return; }
        lb.el.style.display = 'block';
        var rx = (v.x * 0.5 + 0.5) * W;
        var ry = (-v.y * 0.5 + 0.5) * H;
        placeNonOverlapping(lb.el, rx, ry, lb.estW, lb.estH, 16, 68);
      });

      var rScale = (global._replayScale !== undefined) ? global._replayScale : 1.0;
      plantMeshes.forEach(function (pm) {
        if (!pm.labelEl) return;
        var u = pm.data;
        var isPeakDefault = !!defaultCalloutUnits[u.unit];
        var show = isPeakDefault || (selectedUnit === u.unit) || (hoverUnit === u.unit);
        if (!show) { pm.labelEl.style.display = 'none'; return; }
        v.set(pm.group.position.x, 0.64 + 0.44, pm.group.position.z).project(camera);
        if (v.z > 1) { pm.labelEl.style.display = 'none'; return; }
        var effResid = u.resid * rScale;
        var col = mode === 'resid' ? ('#' + residHex(effResid).toString(16).padStart(6, '0')) : LV[u.grade];
        var isHighAlert = (u.grade >= 3 || effResid >= 7.0);
        var cardBg = isHighAlert ? 'rgba(138,19,42,0.92)' : 'rgba(6,20,32,0.90)';
        var tagMetric = (effResid >= 0 ? '+' : '') + effResid.toFixed(2) + '℃';
        var text = u.unit + ' ' + tagMetric;
        pm.labelEl.style.display = 'block';
        pm.labelEl.innerHTML =
          '<div style="display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:5px;background:' + cardBg + ';backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);' +
            'border:1px solid ' + col + ';border-left:3.5px solid ' + col + ';box-shadow:0 3px 12px rgba(0,0,0,0.48);font-family:\'JetBrains Mono\',monospace;font-size:10.5px;font-weight:700;color:#ffffff;">' +
            '<span style="width:6px;height:6px;border-radius:50%;background:' + col + ';box-shadow:0 0 6px ' + col + ';flex-shrink:0;"></span>' +
            '<span>' + text + '</span>' +
          '</div>' +
          '<div class="ldr-line" style="width:1.5px;height:14px;background:' + col + ';margin:0 auto;"></div>' +
          '<div style="width:5px;height:5px;border-radius:50%;background:' + col + ';box-shadow:0 0 6px ' + col + ';margin:-1px auto 0;"></div>';
        var rx = (v.x * 0.5 + 0.5) * W;
        var ry = (-v.y * 0.5 + 0.5) * H;
        placeNonOverlapping(pm.labelEl, rx, ry, 124, 40, 14, 72);
      });
    }

    var flyState = null;
    function flyTo(camPos, lookAt) {
      flyState = {
        pEnd: new THREE.Vector3(camPos[0], camPos[1], camPos[2]),
        tEnd: new THREE.Vector3(lookAt[0], lookAt[1], lookAt[2])
      };
    }

    var raycaster = new THREE.Raycaster();
    var pointer = new THREE.Vector2();
    var downPos = { x: 0, y: 0 };

    canvas.addEventListener('pointerdown', function (e) {
      downPos = { x: e.clientX, y: e.clientY };
      flyState = null;
      canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', function (e) {
      var rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      var hits = raycaster.intersectObjects(pickableObjects, false);
      if (hits.length > 0) {
        var uId = hits[0].object.userData.unit;
        var pm = plantMeshes.filter(function (x) { return x.unit === uId; })[0];
        if (pm) {
          hoverUnit = uId;
          canvas.style.cursor = 'pointer';
          var d = pm.data;
          tip3d.style.display = 'block';
          tip3d.style.left = Math.min(rect.width - 175, e.clientX - rect.left + 14) + 'px';
          tip3d.style.top = Math.max(10, e.clientY - rect.top - 12) + 'px';
          tip3d.innerHTML = d.isCtrl
            ? '<b>' + d.unit + '</b>（对照基准 · 持续供水）<br>W1 持续供水株位 · 冠层热红外对照'
            : '<b>' + d.unit + '</b> · ' + LVN[d.grade] + '胁迫 (' + (d.bed === 'W2' ? 'W2 断水床 A' : 'W3 断水床 B') + ')<br>' +
              '冠层残差温升：<b>' + (d.resid >= 0 ? '+' : '') + d.resid.toFixed(2) + ' ℃</b><br>' +
              '标准化异常 z：' + d.z.toFixed(2);
          updatePlantVisuals();
          return;
        }
      }
      hoverUnit = null;
      tip3d.style.display = 'none';
      if (!e.buttons) canvas.style.cursor = 'grab';
      updatePlantVisuals();
    });
    canvas.addEventListener('pointerup', function (e) {
      canvas.style.cursor = hoverUnit ? 'pointer' : 'grab';
      if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) < 5 && hoverUnit) {
        selectedUnit = hoverUnit;
        updatePlantVisuals();
        var pmHit = plantMeshes.filter(function (x) { return x.unit === hoverUnit; })[0];
        if (pmHit) {
          var gp = pmHit.group.position;
          flyTo([gp.x - 1.35, gp.y + 1.35, gp.z + 1.85], [gp.x, gp.y + 0.25, gp.z]);
        }
        if (rowsEl) {
          var targetRow = Array.from(rowsEl.querySelectorAll('.row')).filter(function (el) {
            return el.querySelector('.row-id') && el.querySelector('.row-id').textContent === hoverUnit;
          })[0];
          if (targetRow) {
            var prev = rowsEl.querySelector('.row.sel'); if (prev) prev.classList.remove('sel');
            targetRow.classList.add('sel');
            targetRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          }
        }
        if (opts && typeof opts.onSelectUnit === 'function') {
          opts.onSelectUnit(pmHit ? pmHit.data : hoverUnit);
        }
      }
    });
    canvas.addEventListener('dblclick', function () {
      flyTo(DEFAULT_CAM_POS, DEFAULT_LOOK_AT);
      selectedUnit = null;
      updatePlantVisuals();
      if (opts && typeof opts.onSelectUnit === 'function') opts.onSelectUnit(null);
    });

    function highlight(unit) {
      selectedUnit = unit;
      updatePlantVisuals();
      var pm = plantMeshes.filter(function (x) { return x.unit === unit; })[0];
      if (pm) {
        var gp = pm.group.position;
        flyTo([gp.x - 1.35, gp.y + 1.35, gp.z + 1.85], [gp.x, gp.y + 0.25, gp.z]);
      }
      if (opts && typeof opts.onSelectUnit === 'function') {
        opts.onSelectUnit(pm ? pm.data : unit);
      }
    }

    function updateLegend() {
      var cbEl = document.getElementById('thermal-colorbar');
      if (cbEl) cbEl.style.display = (mode === 'resid') ? 'flex' : 'none';
      if (!legendBar) return;
      if (mode === 'resid') {
        legendBar.innerHTML =
          '<span><i class="dotc" style="background:#2563eb"></i>&lt;1.8℃ 冷基线</span>' +
          '<span><i class="dotc" style="background:#06b6d4"></i>1.8–3.4℃</span>' +
          '<span><i class="dotc" style="background:#eab308"></i>3.4–5.0℃</span>' +
          '<span><i class="dotc" style="background:#f97316"></i>5.0–6.4℃</span>' +
          '<span><i class="dotc" style="background:#ef4444"></i>&gt;6.4℃ 强热区</span>' +
          '<span style="color:#38bdf8"><i class="dotc" style="background:#38bdf8"></i>W1 对照 (持续供水)</span>';
      } else {
        legendBar.innerHTML =
          '<span><i class="dotc" style="background:#3ddc84"></i>健康</span>' +
          '<span><i class="dotc" style="background:#ffb020"></i>早期轻度</span>' +
          '<span><i class="dotc" style="background:#ff7a45"></i>中度胁迫</span>' +
          '<span><i class="dotc" style="background:#ff4d67"></i>重度胁迫</span>' +
          '<span style="color:#38bdf8"><i class="dotc" style="background:#38bdf8"></i>W1 对照 (持续供水)</span>';
      }
    }

    document.querySelectorAll('.tool').forEach(function (b) {
      b.onclick = function () {
        document.querySelectorAll('.tool').forEach(function (x) { x.classList.remove('on'); });
        b.classList.add('on');
        mode = b.dataset.mode;
        updateLegend();
        updatePlantVisuals();
      };
    });

    var btnRoof = document.getElementById('btn-roof');
    if (btnRoof) btnRoof.onclick = function () {
      roofGroup.visible = !roofGroup.visible;
      sunShaftGroup.visible = roofGroup.visible && (sceneTheme === 'real');
      this.classList.toggle('on', !roofGroup.visible);
    };
    var btnIso = document.getElementById('btn-iso');
    if (btnIso) btnIso.onclick = function () { flyTo(DEFAULT_CAM_POS, DEFAULT_LOOK_AT); };
    var btnTop = document.getElementById('btn-top');
    if (btnTop) btnTop.onclick = function () { flyTo([0.15, 11.2, 0.01], [0.15, 0.6, 0]); };

    var btnTheme = document.getElementById('btn-theme');
    function applySceneTheme(theme) {
      sceneTheme = theme;
      if (sceneTheme === 'real') {
        if (btnTheme) { btnTheme.textContent = '日光物理'; btnTheme.classList.remove('on'); }
        wrapEl.style.background = 'linear-gradient(180deg,#cfe4f2 0%,#e5f0f5 62%,#b9c6cc 100%)';
        outerGround.visible = true; outerGroundMat.color.setHex(0x9b8570); roadGroup.visible = true; skyDome.visible = true; stageGrid.visible = false;
        sunShaftGroup.visible = roofGroup.visible;
        scene.fog.color.setHex(0xd8e6f0); scene.fog.density = 0.0115;
        groundMat.color.setHex(0x9e8872);
        // 暖阳斜射 + 通透天光补光（消除前景钢柱与棚内暗部死黑）
        dirLight.position.set(13.5, 6.8, 7.8);
        dirLight.color.setHex(0xffebd0); dirLight.intensity = 2.45;
        hemiLight.color.setHex(0xb4d6f8); hemiLight.groundColor.setHex(0x7c6852); hemiLight.intensity = 0.68;
        coolFillLight.intensity = 0.78; rimLight.intensity = 0.82;
        renderer.toneMappingExposure = 0.90;
      } else {
        if (btnTheme) { btnTheme.textContent = '暗夜科研'; btnTheme.classList.add('on'); }
        wrapEl.style.background = 'radial-gradient(circle at 50% 42%,#0b2533 0%,#04111b 100%)';
        outerGround.visible = true; outerGroundMat.color.setHex(0x564639); roadGroup.visible = false; skyDome.visible = false; stageGrid.visible = false; sunShaftGroup.visible = false;
        scene.fog.color.setHex(0x04111b); scene.fog.density = 0.022;
        groundMat.color.setHex(0x5c4b3d);
        dirLight.position.set(10.5, 10.2, 6.2);
        dirLight.color.setHex(0xfff0d4); dirLight.intensity = 1.12;
        hemiLight.color.setHex(0x7be8d8); hemiLight.groundColor.setHex(0x1c150e); hemiLight.intensity = 0.36;
        coolFillLight.intensity = 0.42; rimLight.intensity = 0.96;
        renderer.toneMappingExposure = 0.80;
      }
    }
    applySceneTheme(sceneTheme);
    if (btnTheme) btnTheme.onclick = function () {
      applySceneTheme(sceneTheme === 'real' ? 'dark' : 'real');
    };

    /* -------------------- 10. WebGL 实时光学散景景深 (Depth of Field Bokeh) + FXAA 抗锯齿后处理管线 -------------------- */
    var dofEnabled = false; // 默认全场景清晰锐利，不再中间局部聚焦；可通过左下角按钮按需开启景深
    var dofSupported = false;
    var dofRT = null;
    var dofScene = null;
    var dofCamera = null;
    var dofMat = null;

    (function initDoFPipeline() {
      try {
        var gl = renderer.getContext();
        var hasDepthTex = renderer.capabilities.isWebGL2 || !!gl.getExtension('WEBGL_depth_texture');
        if (!hasDepthTex || typeof THREE.DepthTexture !== 'function') return;

        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var initW = Math.max(2, (wrapEl.clientWidth || 800) * dpr);
        var initH = Math.max(2, (wrapEl.clientHeight || 500) * dpr);

        dofRT = new THREE.WebGLRenderTarget(initW, initH, {
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter,
          format: THREE.RGBAFormat
        });
        dofRT.texture.encoding = THREE.sRGBEncoding;
        dofRT.depthTexture = new THREE.DepthTexture();
        dofRT.depthTexture.type = THREE.UnsignedShortType;

        dofMat = new THREE.ShaderMaterial({
          toneMapped: false,
          depthWrite: false,
          depthTest: false,
          uniforms: {
            tColor: { value: dofRT.texture },
            tDepth: { value: dofRT.depthTexture },
            uResolution: { value: new THREE.Vector2(initW, initH) },
            uNear: { value: camera.near },
            uFar: { value: camera.far },
            uFocusDist: { value: 8.5 },
            uAperture: { value: 2.35 },
            uMaxBlur: { value: 4.2 }
          },
          vertexShader: [
            'varying vec2 vUv;',
            'void main() {',
            '  vUv = uv;',
            '  gl_Position = vec4(position.xy, 0.0, 1.0);',
            '}'
          ].join('\n'),
          fragmentShader: [
            'uniform sampler2D tColor;',
            'uniform sampler2D tDepth;',
            'uniform vec2 uResolution;',
            'uniform float uNear;',
            'uniform float uFar;',
            'uniform float uFocusDist;',
            'uniform float uAperture;',
            'uniform float uMaxBlur;',
            'varying vec2 vUv;',
            '',
            'float getLinearDepth(vec2 coord) {',
            '  float fragZ = texture2D(tDepth, coord).x;',
            '  float viewZ = (uNear * uFar) / ((uFar - uNear) * fragZ - uFar);',
            '  return -viewZ;',
            '}',
            '',
            'void main() {',
            '  vec2 texel = 1.0 / uResolution;',
            '  vec4 centerCol = texture2D(tColor, vUv);',
            '  float centerDepth = getLinearDepth(vUv);',
            '',
            '  // 计算物理弥散圆半径 (Circle of Confusion) —— 焦点附件保持锐利，远景与极近前景柔和虚化',
            '  float diff = abs(centerDepth - uFocusDist);',
            '  float deadZone = max(1.35, uFocusDist * 0.22);',
            '  float coc = clamp((max(0.0, diff - deadZone) / max(2.5, uFocusDist)) * uAperture, 0.0, 1.0);',
            '  float blurPx = coc * uMaxBlur;',
            '',
            '  vec3 acc = centerCol.rgb;',
            '  if (blurPx < 0.35) {',
            '    // 焦点清晰区：执行轻量级亚像素边缘抗锯齿与微锐化',
            '    vec3 n = texture2D(tColor, vUv + vec2(0.0, texel.y)).rgb;',
            '    vec3 s = texture2D(tColor, vUv - vec2(0.0, texel.y)).rgb;',
            '    vec3 e = texture2D(tColor, vUv + vec2(texel.x, 0.0)).rgb;',
            '    vec3 w = texture2D(tColor, vUv - vec2(texel.x, 0.0)).rgb;',
            '    vec3 avg4 = 0.25 * (n + s + e + w);',
            '    acc = mix(centerCol.rgb, avg4, 0.16) + (centerCol.rgb - avg4) * 0.08;',
            '  } else {',
            '    // 散景虚化区：12 采样点黄金角 Vogel 螺旋圆盘光斑核 (Golden-Angle Bokeh Kernel)',
            '    float tot = 1.0;',
            '    for (int i = 0; i < 12; i++) {',
            '      float fi = float(i);',
            '      float r = sqrt((fi + 0.5) / 12.0) * blurPx;',
            '      float theta = fi * 2.39996323;',
            '      vec2 offset = vec2(cos(theta), sin(theta)) * r * texel;',
            '      vec2 sUv = clamp(vUv + offset, 0.001, 0.999);',
            '      float sDepth = getLinearDepth(sUv);',
            '      float sDiff = abs(sDepth - uFocusDist);',
            '      float w = (sDepth >= centerDepth - 0.35) ? 1.0 : clamp(sDiff - deadZone, 0.15, 1.0);',
            '      vec3 sCol = texture2D(tColor, sUv).rgb;',
            '      // 高光散景增强 (Bokeh Highlight Boost)',
            '      float luma = dot(sCol, vec3(0.299, 0.587, 0.114));',
            '      w *= (1.0 + max(0.0, luma - 0.72) * 1.4);',
            '      acc += sCol * w;',
            '      tot += w;',
            '    }',
            '    acc /= tot;',
            '  }',
            '',
            '  gl_FragColor = vec4(clamp(acc, 0.0, 1.0), centerCol.a);',
            '}'
          ].join('\n')
        });

        dofScene = new THREE.Scene();
        dofCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        dofScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), dofMat));
        dofSupported = true;
      } catch (err) {
        dofSupported = false;
      }
    })();

    // 灌溉处方抽屉与潮汐补水动画
    (function initRxPanel() {
      var btnRx = document.getElementById('btn-rx');
      var panel = document.getElementById('rx-panel');
      var body = document.getElementById('rx-body');
      var btnClose = document.getElementById('rx-close');
      var btnGo = document.getElementById('rx-irrigate');
      var btnRst = document.getElementById('rx-reset');
      if (!btnRx || !panel) return;
      var simTimer = null;

      function bedStats(bed) {
        var list = U.filter(function (u) { return u.bed === bed; });
        var avg = list.reduce(function (s, u) { return s + u.resid; }, 0) / Math.max(1, list.length);
        var l3 = list.filter(function (u) { return u.grade >= 3; }).length;
        return { avg: avg, l3: l3 };
      }
      function renderRx(isIrrigated) {
        if (isIrrigated) {
          body.innerHTML =
            '<div class="rx-item ok"><b>W1 对照床</b> · 持续供水对照基准（不参与判级）</div>' +
            '<div class="rx-item ok"><b>W2 / W3 断水床</b> · 模拟复水回落演示完成<br>按 ADA2025 实测 79.9% 复水期平均回落幅度推演，注意复水后仍略高于基线（部分恢复）。</div>';
          return;
        }
        var w2 = bedStats('W2'), w3 = bedStats('W3');
        body.innerHTML =
          '<div class="rx-item ' + (w3.l3 > 0 ? 'sev' : '') + '"><b>W3 断水床 B（实测响应更强）</b><br>窗末均值 <b>+' + w3.avg.toFixed(2) + '℃</b>，L3 重度 <b>' + w3.l3 + '</b> 株；建议优先纳入复水观察与热红外复核。</div>' +
          '<div class="rx-item"><b>W2 断水床 A（实测响应较缓）</b><br>窗末均值 <b>+' + w2.avg.toFixed(2) + '℃</b>，L3 重度 <b>' + w2.l3 + '</b> 株；建议同步开展复水后回落追踪。</div>' +
          '<div class="rx-item ok"><b>W1 对照床</b> · 持续供水基准（共 11 株位）</div>';
      }
      renderRx(false);
      btnRx.onclick = function () { panel.classList.toggle('on'); btnRx.classList.toggle('on'); };
      if (btnClose) btnClose.onclick = function () { panel.classList.remove('on'); btnRx.classList.remove('on'); };
      if (btnGo) btnGo.onclick = function () {
        if (simTimer) clearInterval(simTimer);
        ['W2', 'W3'].forEach(function (b) { waterSheets[b].mesh.visible = true; waterSheets[b].mat.opacity = 0.52; });
        var step = 0;
        simTimer = setInterval(function () {
          step++;
          global._replayScale = 1.0 - Math.min(1, step / 20) * 0.799;
          updatePlantVisuals();
          if (step >= 20) { clearInterval(simTimer); renderRx(true); }
        }, 55);
      };
      if (btnRst) btnRst.onclick = function () {
        if (simTimer) clearInterval(simTimer);
        ['W2', 'W3'].forEach(function (b) { waterSheets[b].mesh.visible = false; waterSheets[b].mat.opacity = 0.0; });
        global._replayScale = 1.0;
        updatePlantVisuals();
        renderRx(false);
      };
    })();

    // 主渲染动画循环
    var scanPhase = 0;
    function animate() {
      requestAnimationFrame(animate);
      var W = wrapEl.clientWidth || 800;
      var H = wrapEl.clientHeight || 500;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.floor(W * dpr) || canvas.height !== Math.floor(H * dpr)) {
        renderer.setSize(W, H, false);
        camera.aspect = W / H;
        camera.updateProjectionMatrix();
        if (dofSupported && dofRT) {
          dofRT.setSize(Math.max(2, Math.floor(W * dpr)), Math.max(2, Math.floor(H * dpr)));
          dofMat.uniforms.uResolution.value.set(Math.floor(W * dpr), Math.floor(H * dpr));
        }
      }
      if (flyState) {
        camera.position.lerp(flyState.pEnd, 0.08);
        controls.target.lerp(flyState.tEnd, 0.08);
        if (camera.position.distanceTo(flyState.pEnd) < 0.05) flyState = null;
      }
      scanPhase += 0.007;
      gantryGroup.position.x = Math.sin(scanPhase) * 4.4;
      if (global._w1FlowDrops) {
        global._w1FlowDrops.forEach(function (d) {
          var t = (d.phase + scanPhase * 0.6) % 1.0;
          d.mesh.position.x = -4.8 + t * 9.8;
        });
      }
      fanBlades.forEach(function (fb) { fb.rotation.x += 0.08; });
      innerFans.forEach(function (ib) { ib.rotation.x += 0.11; });
      // 漂浮微气候体积水雾层缓慢流动
      mistLayers.forEach(function (ml) {
        ml.mesh.position.x = ml.baseX + Math.sin(scanPhase * 1.4 + ml.phase) * 0.45;
        ml.mesh.position.y = ml.baseY + Math.cos(scanPhase * 1.1 + ml.phase) * 0.06;
      });
      if (mode === 'resid') {
        plantMeshes.forEach(function (pm, idx) {
          if (!pm.pulseMesh.visible) return;
          var s = 1.0 + 0.16 * Math.sin(scanPhase * 5.0 + idx * 0.45);
          pm.pulseMesh.scale.set(s, 1, s);
        });
      }
      controls.update();

      if (dofSupported && dofEnabled && dofRT) {
        // 焦点距离实时锁定当前相机观察目标 (controls.target)，使聚焦植株始终锐利、远近景自然散景虚化
        dofMat.uniforms.uFocusDist.value = camera.position.distanceTo(controls.target);
        renderer.setRenderTarget(dofRT);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        renderer.render(dofScene, dofCamera);
      } else {
        renderer.setRenderTarget(null);
        renderer.render(scene, camera);
      }
      updateLabels(W, H);
    }
    animate();

    function setLayer(name, visible) {
      if (name === 'roof') {
        roofGroup.visible = !!visible;
        sunShaftGroup.visible = !!visible && (sceneTheme === 'real');
        if (btnRoof) btnRoof.classList.toggle('on', !visible);
      } else if (name === 'glass') {
        glassMeshes.forEach(function (gm) { gm.visible = !!visible; });
        var bgEl = document.getElementById('btn-glass');
        if (bgEl) bgEl.classList.toggle('on', !!visible);
      } else if (name === 'labels') {
        labelsEl.style.display = visible ? '' : 'none';
      } else if (name === 'soil') {
        outerGround.visible = !!visible;
        if (plotPlinth) plotPlinth.visible = !!visible;
        if (plotRim) plotRim.visible = !!visible;
      } else if (name === 'gantry') {
        gantryGroup.visible = !!visible;
      }
    }

    function setTimelineScale(scale, showWater) {
      global._replayScale = Math.max(0.0, Math.min(1.35, Number(scale)));
      ['W2', 'W3'].forEach(function (b) {
        if (waterSheets[b]) {
          waterSheets[b].mesh.visible = !!showWater;
          waterSheets[b].mat.opacity = showWater ? 0.46 : 0.0;
        }
      });
      updatePlantVisuals();
    }

    return { highlight: highlight, setLayer: setLayer, setTimelineScale: setTimelineScale };
  };
})(window);
