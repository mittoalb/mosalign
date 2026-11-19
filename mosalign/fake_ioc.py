"""Fake IOC for testing mosalign without hardware.

Serves every PV mosalign touches, under one prefix (default ``mosfake:``),
and prints each write so you can watch what mosalign does:

  CA   <P>m1, <P>m2                       motor records; .RBV ramps to VAL
  CA   <P>TomoScan:ExposureTime, NumAngles, RotationStart/Step/Stop,
       FileName, StartScan, AbortScan, ScanStatus, ServerRunning
  CA   <P>TXMOptics:CropLeft/Right/Top/Bottom, Crop
  CA   <P>cam1:MaxSizeX_RBV, MaxSizeY_RBV
  PVA  <P>Pva1:Image                      NTNDArray: a synthetic sample seen
                                          at the current m1/m2 position

StartScan behaves like TomoScan's busy record: `caput -c` returns when the
simulated scan ends (NumAngles x ExposureTime / --speedup seconds) or when
AbortScan is written.

Run:   mosalign-fake-ioc            (or: python -m mosalign.fake_ioc)
Then in mosalign set
  Motor 1 PV      mosfake:m1          Motor 2 PV   mosfake:m2
  Image PV        mosfake:Pva1:Image  TomoScan     mosfake:TomoScan:
  Camera prefix   mosfake:cam1        Crop prefix  mosfake:TXMOptics
For production, put the real PV names back in the same fields.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty, run

STATE = {"m1": 0.0, "m2": 0.0, "m3": 0.0,   # live motor readbacks (mm, m3 deg)
         "single": False,     # camera ImageMode Single: frames only on trigger
         "trigger": 0,        # bumped once per triggered frame
         "acquiring": False}


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def logged(name, value, **kw):
    """A pvproperty that prints every write."""
    prop = pvproperty(name=name, value=value, **kw)

    @prop.putter
    async def prop(group, instance, val):
        _log(f"put {instance.pvname} = {val}")
        return val
    return prop


def motor(name):
    prop = pvproperty(name=name, value=0.0, record="motor", precision=4)

    @prop.putter
    async def prop(group, instance, target):
        _log(f"move {instance.pvname} → {target:.4f} mm "
             f"(from {STATE[name]:.4f})")
        group.start_move(name, instance, float(target))
        # Like a real motor record: put-completion (caput -c / wait=True)
        # fires only when the move has finished.
        await group._moves[name]
        return target
    return prop


class FakeBeamline(PVGroup):
    m1 = motor("m1")
    m2 = motor("m2")
    m3 = motor("m3")      # rotation

    # TomoScan's sample motors are m1/m2 (names written at startup). After
    # flat fields TomoScan moves the sample to SampleInX/Y — emulated in
    # start_scan, so a scanner that forgets to set them is caught.
    sample_x_pvname = pvproperty(name="TomoScan:SampleXPVName", value="",
                                 dtype=ChannelType.STRING)
    sample_y_pvname = pvproperty(name="TomoScan:SampleYPVName", value="",
                                 dtype=ChannelType.STRING)
    sample_in_x = logged("TomoScan:SampleInX", 0.0, precision=4)
    sample_in_y = logged("TomoScan:SampleInY", 0.0, precision=4)
    flat_mode = pvproperty(name="TomoScan:FlatFieldMode", value="Start",
                           dtype=ChannelType.STRING)

    # Remaining PVs the `tomoscan` CLI (bin/tomoscan init_PVs) connects
    # to, so the real CLI can be run against this IOC.
    _s = dict(dtype=ChannelType.STRING)
    scan_type = logged("TomoScan:ScanType", "Single", **_s)
    flat_axis = pvproperty(name="TomoScan:FlatFieldAxis", value="Both", **_s)
    file_plugin = pvproperty(name="TomoScan:FilePluginPVPrefix", value="", **_s)
    sample_name = pvproperty(name="TomoScan:SampleName", value="sample", **_s)
    file_template = pvproperty(name="HDF1:FileTemplate", value="%s%s_%3.3d.h5", **_s)
    file_number = pvproperty(name="HDF1:FileNumber", value=1)
    dark_mode = pvproperty(name="TomoScan:DarkFieldMode", value="Start", **_s)
    dark_value = pvproperty(name="TomoScan:DarkFieldValue", value=0.0)
    flat_value = pvproperty(name="TomoScan:FlatFieldValue", value=0.0)
    diff_flat_exp = pvproperty(name="TomoScan:DifferentFlatExposure", value=0)
    flat_exp = pvproperty(name="TomoScan:FlatExposureTime", value=0.1)
    flip_stitch = pvproperty(name="TomoScan:FlipStitch", value=0)
    n_dark = pvproperty(name="TomoScan:NumDarkFields", value=10)
    n_flat = pvproperty(name="TomoScan:NumFlatFields", value=10)
    return_rot = pvproperty(name="TomoScan:ReturnRotation", value=1)
    out_angle = pvproperty(name="TomoScan:SampleOutAngle", value=0.0)
    out_angle_en = pvproperty(name="TomoScan:SampleOutAngleEnable", value=0)
    out_x = pvproperty(name="TomoScan:SampleOutX", value=3.0)
    out_y = pvproperty(name="TomoScan:SampleOutY", value=0.0)

    @file_plugin.startup
    async def file_plugin(self, instance, async_lib):
        await instance.write(self.prefix + "HDF1:")

    @sample_x_pvname.startup
    async def sample_x_pvname(self, instance, async_lib):
        await instance.write(self.prefix + "m1")

    @sample_y_pvname.startup
    async def sample_y_pvname(self, instance, async_lib):
        await instance.write(self.prefix + "m2")

    # Camera (single-image acquisition) and XANES energy range PVs.
    acquire_time = logged("cam1:AcquireTime", 0.1, precision=3)
    image_mode = pvproperty(name="cam1:ImageMode", value=2)   # 0 Single, 2 Continuous

    @image_mode.putter
    async def image_mode(self, instance, value):
        _log(f"put {instance.pvname} = {value}")
        STATE["single"] = (int(value) == 0)
        return value
    num_images = logged("cam1:NumImages", 1)
    shutter = logged("shutter", 0)            # fast shutter stand-in

    # B-station shutter stand-in: write 1 to open/close, status 1 = closed
    # (like PB:32ID:STA_B_SBS_CLSD_PL), reached after a 1.5 s stroke.
    b_open = pvproperty(name="bshutter:open", value=0)
    b_close = pvproperty(name="bshutter:close", value=0)
    b_closed = pvproperty(name="bshutter:closed", value=0)

    @b_open.putter
    async def b_open(self, instance, value):
        _log("B SHUTTER open requested")
        await asyncio.sleep(1.5)
        await self.b_closed.write(0)
        _log("B SHUTTER open")
        return 0

    @b_close.putter
    async def b_close(self, instance, value):
        _log("B SHUTTER close requested")
        await asyncio.sleep(1.5)
        await self.b_closed.write(1)
        _log("B SHUTTER closed")
        return 0
    acquire = pvproperty(name="cam1:Acquire", value=0)
    xanes_start = logged("TXMOptics:XanesStart", 10.2, precision=4)
    xanes_end = logged("TXMOptics:XanesEnd", 10.3, precision=4)
    xanes_step = logged("TXMOptics:XanesStep", 10.0)

    @acquire.putter
    async def acquire(self, instance, value):
        """Like ADSpinnaker in Single mode: Acquire reads 1 while the frame is
        being taken; a trigger while busy is ignored; the frame reaches the
        PVA stream ~0.3 s BEFORE Acquire goes back to 0."""
        if not value:
            return 0
        if STATE["acquiring"]:
            _log("ACQUIRE ignored — camera still busy")
            return 1
        STATE["acquiring"] = True
        _log(f"ACQUIRE image at X={STATE['m1']:.4f} Y={STATE['m2']:.4f} "
             f"rot={STATE['m3']:.2f}  ({self.acquire_time.value} s)")

        async def expose():
            try:
                await asyncio.sleep(float(self.acquire_time.value))
                STATE["trigger"] += 1          # frame published now…
                await asyncio.sleep(0.3)       # …driver done a bit later
            finally:
                STATE["acquiring"] = False
                await self.acquire.write(0)
        asyncio.get_event_loop().create_task(expose())
        return 1

    exposure = logged("TomoScan:ExposureTime", 0.1, precision=3)
    num_angles = logged("TomoScan:NumAngles", 1500)
    rot_start = logged("TomoScan:RotationStart", 0.0)
    rot_step = logged("TomoScan:RotationStep", 0.12)
    rot_stop = logged("TomoScan:RotationStop", 180.0)
    file_name = logged("TomoScan:FileName", "test", dtype=ChannelType.STRING)
    scan_status = pvproperty(name="TomoScan:ScanStatus", value="Scan complete",
                             dtype=ChannelType.STRING)
    server_running = pvproperty(name="TomoScan:ServerRunning", value=1)

    crop_left = logged("TXMOptics:CropLeft", 0)
    crop_right = logged("TXMOptics:CropRight", 0)
    crop_top = logged("TXMOptics:CropTop", 0)
    crop_bottom = logged("TXMOptics:CropBottom", 0)
    crop = logged("TXMOptics:Crop", 0)

    start_scan = pvproperty(name="TomoScan:StartScan", value=0)
    abort_scan = pvproperty(name="TomoScan:AbortScan", value=0)

    def __init__(self, *a, frame_w, frame_h, velocity, speedup, **kw):
        super().__init__(*a, **kw)
        self.velocity = velocity
        self.speedup = speedup
        self._moves = {}
        self._abort = asyncio.Event()
        self.frame_w, self.frame_h = frame_w, frame_h

    max_x = pvproperty(name="cam1:MaxSizeX_RBV", value=0, read_only=True)
    max_y = pvproperty(name="cam1:MaxSizeY_RBV", value=0, read_only=True)

    @max_x.startup
    async def max_x(self, instance, async_lib):
        await instance.write(self.frame_w)

    @max_y.startup
    async def max_y(self, instance, async_lib):
        await instance.write(self.frame_h)

    # ── motors ──────────────────────────────────────────────────────────

    def start_move(self, name, instance, target):
        old = self._moves.get(name)
        if old and not old.done():
            old.cancel()
        self._moves[name] = asyncio.get_event_loop().create_task(
            self._ramp(name, instance, target))

    async def _ramp(self, name, instance, target):
        rbv, dmov = instance.field_inst.user_readback_value, \
            instance.field_inst.done_moving_to_value
        await dmov.write(0)
        dt = 0.05
        step = self.velocity * dt
        while abs(STATE[name] - target) > 1e-9:
            d = target - STATE[name]
            STATE[name] = target if abs(d) <= step else STATE[name] + np.sign(d) * step
            await rbv.write(STATE[name])
            await asyncio.sleep(dt)
        await dmov.write(1)
        _log(f"  {instance.pvname} at {target:.4f} mm")

    # ── TomoScan busy record ───────────────────────────────────────────

    @start_scan.putter
    async def start_scan(self, instance, value):
        if not value:
            return 0
        n, expo = int(self.num_angles.value), float(self.exposure.value)
        dur = n * expo / self.speedup
        _log(f"START SCAN  X={STATE['m1']:.4f} Y={STATE['m2']:.4f}  "
             f"NumAngles={n} ExposureTime={expo}  "
             f"crop L{self.crop_left.value} R{self.crop_right.value} "
             f"T{self.crop_top.value} B{self.crop_bottom.value}  "
             f"file={self.file_name.value!r}  ({dur:.1f}s simulated)")
        self._abort.clear()
        await self.scan_status.write("Scanning")
        # Flats, then back to SampleIn — where the projections really are.
        tx, ty = float(self.sample_in_x.value), float(self.sample_in_y.value)
        if abs(tx - STATE["m1"]) > 1e-4 or abs(ty - STATE["m2"]) > 1e-4:
            _log(f"  ⚠ SampleIn X={tx:.4f} Y={ty:.4f} differs from the current "
                 f"position — TomoScan moves the sample there after flats")
        for name, inst, tgt in (("m1", self.m1, tx), ("m2", self.m2, ty)):
            self.start_move(name, inst, tgt)
        await asyncio.gather(*(t for t in self._moves.values() if not t.done()))
        _log(f"  projections at X={STATE['m1']:.4f} Y={STATE['m2']:.4f}")
        try:
            await asyncio.wait_for(self._abort.wait(), timeout=dur)
            _log("SCAN ABORTED")
            await self.scan_status.write("Aborted")
        except asyncio.TimeoutError:
            _log("scan done")
            await self.scan_status.write("Scan complete")
        return 0     # put completion fires now → `caput -c` returns

    @abort_scan.putter
    async def abort_scan(self, instance, value):
        _log(f"put {instance.pvname} = {value}")
        if value:
            self._abort.set()
        return 0


# ── synthetic camera on pvAccess ───────────────────────────────────────

def _sample(xmm, ymm):
    """Sample intensity at absolute stage coordinates (mm): a 0.1 mm grid,
    heavier lines every 0.5 mm, and fixed random blobs, so a misplaced
    frame shows up as a broken line in the mosaic."""
    img = 2000 + 300 * np.sin(xmm * 3.0) * np.cos(ymm * 2.0)
    for period, width, amp in ((0.1, 0.003, 4000), (0.5, 0.008, 8000)):
        on_x = np.abs((xmm + period / 2) % period - period / 2) < width
        on_y = np.abs((ymm + period / 2) % period - period / 2) < width
        img = img + amp * (on_x | on_y)
    x0, x1, y0, y1 = xmm.min(), xmm.max(), ymm.min(), ymm.max()
    for bx, by, r in _BLOBS:
        if not (x0 - 4 * r < bx < x1 + 4 * r and y0 - 4 * r < by < y1 + 4 * r):
            continue
        img = img + 15000 * np.exp(-((xmm - bx) ** 2 + (ymm - by) ** 2) / (2 * r * r))
    return np.clip(img, 0, 65535).astype(np.uint16)


_rng = np.random.default_rng(0)
_BLOBS = [(x, y, r) for x, y, r in zip(_rng.uniform(-5, 5, 400),
                                        _rng.uniform(-5, 5, 400),
                                        _rng.uniform(0.01, 0.04, 400))]


def serve_image(pv_name, w, h, pixel_um, rate_hz):
    import pvaccess as pva
    px_mm = pixel_um / 1000.0
    # Row index increases with m2, column with m1 — the same orientation
    # mosalign's grid stitcher assumes.
    cols = (np.arange(w) - w / 2) * px_mm
    rows = (np.arange(h) - h / 2) * px_mm

    def frame(uid):
        a = pva.NtNdArray()
        X = STATE["m1"] + cols[None, :]
        Y = STATE["m2"] + rows[:, None]
        a["value"] = {"ushortValue": _sample(X, Y).ravel()}
        dim = lambda n: {"size": n, "offset": 0, "fullSize": n,
                         "binning": 1, "reverse": False}
        a["dimension"] = [dim(w), dim(h)]
        a["uniqueId"] = uid
        return a

    server = pva.PvaServer(pv_name, frame(0))
    uid, seen = 0, STATE["trigger"]
    while True:
        if STATE["single"]:            # Single mode: one frame per trigger
            if STATE["trigger"] != seen:
                seen = STATE["trigger"]
                uid += 1
                server.update(pv_name, frame(uid))
            time.sleep(0.01)
            continue
        uid += 1                       # Continuous: free-running stream
        server.update(pv_name, frame(uid))
        time.sleep(1.0 / rate_hz)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="mosfake:")
    p.add_argument("--width", type=int, default=800, help="frame width, px")
    p.add_argument("--height", type=int, default=600, help="frame height, px")
    p.add_argument("--pixel-size", type=float, default=0.766, help="µm per px")
    p.add_argument("--velocity", type=float, default=1.0, help="motor speed, mm/s")
    p.add_argument("--speedup", type=float, default=50.0,
                   help="simulated scan = NumAngles*ExposureTime/speedup s")
    p.add_argument("--rate", type=float, default=5.0, help="image frames per s")
    args = p.parse_args()

    img_pv = f"{args.prefix}Pva1:Image"
    threading.Thread(target=serve_image, daemon=True,
                     args=(img_pv, args.width, args.height,
                           args.pixel_size, args.rate)).start()

    ioc = FakeBeamline(prefix=args.prefix, frame_w=args.width,
                       frame_h=args.height, velocity=args.velocity,
                       speedup=args.speedup)
    _log(f"fake IOC up — prefix {args.prefix!r}, image {img_pv} "
         f"{args.width}×{args.height} @ {args.pixel_size} µm/px")
    for name in sorted(ioc.pvdb):
        print("   ", name)
    run(ioc.pvdb, interfaces=["0.0.0.0"], log_pv_names=False)


if __name__ == "__main__":
    main()
