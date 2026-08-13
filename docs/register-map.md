# AXI-Lite register map — `mlp_controller`

Derived from the user logic at the end of
[`ip/mlp_controller_1_0/hdl/mlp_controller_slave_lite_v1_0_S00_AXI.v`](../ip/mlp_controller_1_0/hdl/mlp_controller_slave_lite_v1_0_S00_AXI.v).
The slave is 32 bits wide with an 8-bit address space (64 registers).

| Offset | Register | Access | Contents |
|---|---|---|---|
| `0x00` | `slv_reg0` | RW | Unused |
| `0x04`–`0x88` | `slv_reg1`–`slv_reg34` | W | Feature bytes. Bits `[7:0]` of each word are concatenated into the 272-bit `movenet_data` bus, `slv_reg1` at the LSB. |
| `0x8C` | `slv_reg35` | W | Start. Writing bit 0 asserts `movenet_data_valid` for one clock, launching the FSM from `IDLE`. |
| `0x90` | `slv_reg36` | R | Done. Set when the FSM reaches `DONE`; held for 3 cycles by a counter so the PS can observe it. |
| `0x94` | `slv_reg37` | R | Class, bits `[1:0]`. Latched with the done pulse and retained until the next inference. |
| `0x98`+ | `slv_reg38`+ | — | Present in the AXI template, unused by the design. |

## Feature encoding

34 features: 17 keypoints × (y, x), confidence dropped. Ordering is
`[y0, x0, y1, x1, … y16, x16]`, each an 8-bit **signed** value consumed by the
DSP as `signed [7:0]`.

Normalized `[0, 1]` coordinates should be scaled by 127 and written as a
sign-extended byte in the low bits of each register.

> Note: the original PS controller scaled by 255 and clipped to `[0, 255]`, which
> maps the upper half of the coordinate range onto negative weights once the DSP
> interprets the byte as signed. `software/pulse_monitor.py` now scales by 127.

## Class encoding

| Value | Meaning |
|---|---|
| `2'b00` | Push-up |
| `2'b01` | Squat |
| `2'b10` | Curl |
| `2'b11` | No pose — emitted when no output neuron strictly dominates the other two |

## Handshake

```
PS                                     PL
──                                     ──
write 0x04..0x88  (34 feature bytes)
write 0x8C = 1    ────────────────────▶ movenet_data_valid, IDLE → LAYER1_BIAS
write 0x8C = 0                          … ~100k cycles of MAC sequencing …
poll  0x90        ◀──────────────────── done_pulse, held 3 cycles
read  0x94        ◀──────────────────── pose_class[1:0]
                                        → IDLE
```

There is no interrupt line; the PS polls. At roughly 1 ms per inference against a
100 ms frame budget, polling costs nothing worth optimizing.
