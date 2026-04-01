# STDF Parser Expansion Guide

## What is STDF?

**STDF (Standard Test Data Format)** is the semiconductor industry standard for test data. Every chip on every wafer goes through automated test equipment (ATE) that produces STDF files. These binary files contain everything: lot info, wafer maps, die coordinates, bin results, and parametric measurements.

If you work in semiconductor test data, you will encounter STDF files from Teradyne, Advantest, and other ATE vendors.

---

## Complete STDF V4 Record Types

The full spec defines **26+ record types** across 7 groups:

### File Info Records
| Code | Record | Purpose |
|------|--------|---------|
| (0, 10) | **FAR** | File attributes — STDF version and **endianness** (must be parsed first) |
| (0, 20) | **ATR** | Audit trail — modification timestamps |
| (0, 30) | **VUR** | Version update (V4-2007 extension) |

### Per-Lot Records
| Code | Record | Purpose |
|------|--------|---------|
| (1, 10) | **MIR** | Master info — lot_id, device_name, facility, flow, temperature, test program |
| (1, 20) | **MRR** | Master results — lot finish time, disposition |
| (1, 30) | **PCR** | Part count — total tested, passed, failed, retested per head/site |
| (1, 40) | **HBR** | Hardware bin definitions — bin number → name + pass/fail |
| (1, 50) | **SBR** | Software bin definitions — same as HBR but software-assigned |
| (1, 60) | **PMR** | Pin map — channel/pin index to name mapping |
| (1, 62) | **PGR** | Pin group — groups of pins (e.g., data bus, address bus) |
| (1, 63) | **PLR** | Pin list — ordered pin sequence |
| (1, 70) | **RDR** | Retest data — which bins get retested |
| (1, 80) | **SDR** | Site description — handler, DIB, cables, contacts |

### Per-Wafer Records
| Code | Record | Purpose |
|------|--------|---------|
| (2, 10) | **WIR** | Wafer info — wafer_id, start time |
| (2, 20) | **WRR** | Wafer results — finish time, part counts, yield |
| (2, 30) | **WCR** | Wafer config — wafer diameter, die size, flat/notch orientation, center X/Y |

### Per-Die Records
| Code | Record | Purpose |
|------|--------|---------|
| (5, 10) | **PIR** | Part info — marks the start of testing for one die |
| (5, 20) | **PRR** | Part results — X/Y coordinates, hard_bin, soft_bin, pass/fail, num_tests |

### Per-Test-Execution Records (highest volume)
| Code | Record | Purpose |
|------|--------|---------|
| (10, 30) | **TSR** | Test synopsis — per-test pass/fail counts, exec time, min/max values |
| (15, 10) | **PTR** | Parametric test result — single measurement value + limits + units |
| (15, 15) | **MPR** | Multi-result parametric — array of values per pin (IDDQ, leakage) |
| (15, 20) | **FTR** | Functional test — digital pattern pass/fail + failing pin mask |
| (15, 30) | **STR** | Scan test (V4-2007) — scan chain diagnosis |

### Program Section Records
| Code | Record | Purpose |
|------|--------|---------|
| (20, 10) | **BPS** | Begin program section |
| (20, 20) | **EPS** | End program section |

### Generic Records
| Code | Record | Purpose |
|------|--------|---------|
| (50, 10) | **GDR** | Generic data — vendor-specific custom fields |
| (50, 30) | **DTR** | Datalog text — free-form debug/notes |

---

## What Our Current Parser Handles (P4)

The parser in `src/data/stdf_parser.py` currently handles a **minimal subset**:

| Record | Status | Notes |
|--------|--------|-------|
| FAR | Defined, not parsed | Endianness assumed little-endian |
| MIR | Partial | Lot ID extracted via byte slice (fragile) |
| WIR | Partial | Reads HEAD_NUM only, misses WAFER_ID |
| PRR | Partial | Has **known offset issues** — X/Y coords may be misread |
| PTR | Stub | Record type recognized but no data extracted |

This is sufficient for the current project goal (wafer map → image classification) because we primarily need die coordinates (PRR) and bin results to construct a wafer map image.

---

## Expansion Roadmap

### Priority 1: Fix Existing Records (HIGH IMPACT, LOW EFFORT)

**1a. Add FAR parsing for endianness detection**
```python
def _parse_far(self, data: bytes):
    cpu_type = data[0]  # 1=big-endian, 2=little-endian
    stdf_ver = data[1]  # Should be 4
    self.endian = '>' if cpu_type == 1 else '<'
```
Without this, the parser **silently fails** on big-endian files from certain ATE vendors.

**1b. Fix PRR field offsets**
The STDF V4 PRR layout is:
```
Offset 0: HEAD_NUM (U1)
Offset 1: SITE_NUM (U1)
Offset 2: PART_FLG (B1) — bit flags for pass/fail/abnormal
Offset 3: NUM_TEST (U2) — number of tests executed
Offset 5: HARD_BIN (U2)
Offset 7: SOFT_BIN (U2)
Offset 9: X_COORD  (I2) — signed 16-bit
Offset 11: Y_COORD (I2) — signed 16-bit
```
Current parser reads X_COORD at offset 0 — this is wrong.

**1c. Fix MIR to use proper Cn field parsing**
MIR has 12 bytes of fixed fields before LOT_ID:
```
SETUP_T (U4), START_T (U4), STAT_NUM (U1), MODE_COD (C1), RTST_COD (C1), PROT_COD (C1), BURN_TIM (U2), CMOD_COD (C1)
= 14 bytes
Then LOT_ID as Cn (length byte + string)
```

### Priority 2: Add High-Value Records (HIGH IMPACT)

**2a. HBR + SBR — Bin Definitions**
Without these, bin numbers are meaningless numbers. With them, you can map bin 1 → "Pass", bin 5 → "Leakage Fail", etc.
```python
def _parse_hbr(self, data: bytes):
    head_num = data[0]
    site_num = data[1]
    hbin_num = struct.unpack(self.endian + 'H', data[2:4])[0]
    hbin_cnt = struct.unpack(self.endian + 'I', data[4:8])[0]
    hbin_pf = chr(data[8])  # 'P' or 'F'
    hbin_nam_len = data[9]
    hbin_nam = data[10:10+hbin_nam_len].decode('ascii')
```

**2b. WCR — Wafer Configuration**
Required for accurate wafer map rendering:
```
WAFR_SIZ (R4) — wafer diameter in mm (200 or 300)
DIE_HT   (R4) — die height in mm
DIE_WID  (R4) — die width in mm
WF_UNITS (U1) — 1=inches, 2=cm, 3=mm, 4=mils
WF_FLAT  (C1) — flat orientation: U/D/L/R
CENTER_X (I2) — center die X coordinate
CENTER_Y (I2) — center die Y coordinate
```
This lets you reconstruct the physical wafer layout with correct scale.

**2c. PTR — Parametric Test Results**
This is the **bulk of any STDF file** (often 90%+ of all records). Each PTR is one measurement:
```
TEST_NUM  (U4)  — test number
HEAD_NUM  (U1)
SITE_NUM  (U1)
TEST_FLG  (B1)  — pass/fail/alarm flags
PARM_FLG  (B1)  — limit flags
RESULT    (R4)  — the actual measurement value
TEST_TXT  (Cn)  — test name
ALARM_ID  (Cn)  — alarm identifier
OPT_FLAG  (B1)  — which optional fields are present
RES_SCAL  (I1)  — result scaling exponent
LLM_SCAL  (I1)  — low limit scaling exponent
HLM_SCAL  (I1)  — high limit scaling exponent
LO_LIMIT  (R4)  — low spec limit
HI_LIMIT  (R4)  — high spec limit
UNITS     (Cn)  — measurement units (V, A, Ohm, Hz, etc.)
```

### Priority 3: Add Supporting Records (MEDIUM IMPACT)

| Record | Use Case |
|--------|----------|
| **TSR** | Per-test statistics — avoids re-aggregating thousands of PTRs |
| **PCR** | Part counts per wafer — quick yield calculation |
| **WRR** | Wafer-level results — finish time, overall pass/fail/retest counts |
| **FTR** | Functional test results — digital pattern failures |
| **SDR** | Site config — needed for multi-site testing environments |

### Priority 4: Advanced Records (LOW PRIORITY)

| Record | Use Case |
|--------|----------|
| **MPR** | Multi-pin parametric — IDDQ testing, pin-level leakage |
| **PMR/PGR/PLR** | Pin mapping — needed only for pin-level analysis |
| **PSR/NMR/CNR/SSR/CDR/STR** | Scan diagnosis — V4-2007 extension for DFT/scan chains |
| **GDR/DTR** | Vendor custom data — varies by ATE vendor |

---

## STDF Binary Format Basics

Every record follows this structure:
```
[REC_LEN: U2][REC_TYP: U1][REC_SUB: U1][...data...]
```

Key data types:
| Type | Meaning | Size |
|------|---------|------|
| U1 | Unsigned 1 byte | 1 |
| U2 | Unsigned 2 bytes | 2 |
| U4 | Unsigned 4 bytes | 4 |
| I1 | Signed 1 byte | 1 |
| I2 | Signed 2 bytes | 2 |
| R4 | 32-bit float | 4 |
| C1 | Single character | 1 |
| Cn | Length-prefixed string | 1 + N |
| Bn | Length-prefixed bytes | 1 + N |
| B1 | Bit-encoded byte | 1 |
| Dn | Bit count + bit data | 2 + ceil(N/8) |
| Vn | Variable type | varies |

**Critical**: Cn strings have a **length byte** followed by that many ASCII characters. Example: `\x05HELLO` = "HELLO".

---

## Reference Implementation

The P5 project (`Wafer-Yield-Intelligence`) has a **complete generic STDF parser** in `stdf_reader.py` that:
- Reads all record types from `stdf_v4.json` spec definition
- Handles endianness from FAR
- Properly unpacks all Cn, Dn, Bn, Vn, Kn types
- Supports V4-2007 extensions
- Handles array multipliers (INDX_CNT, RTN_ICNT, etc.)

This can serve as a reference when expanding the P4 parser.

---

## Typical STDF File Structure

```
FAR                          ← Always first (endianness)
MIR                          ← Lot metadata
SDR                          ← Site config
PMR × N                      ← Pin definitions
  WIR                        ← Wafer start
  WCR                        ← Wafer dimensions
    PIR                      ← Die start
      PTR × 50-500           ← Parametric tests for this die
      FTR × 10-100           ← Functional tests for this die
    PRR                      ← Die result (X/Y, bin, P/F)
    ... (repeat PIR→PRR for each die, 1000-50000 times)
  WRR                        ← Wafer summary
  ... (repeat WIR→WRR for each wafer, 25 times)
HBR × N                      ← Hardware bin summary
SBR × N                      ← Software bin summary
PCR                           ← Part counts
TSR × N                      ← Test statistics
MRR                          ← Lot end
```

A typical lot with 25 wafers × 5000 dies × 200 tests = **25 million PTR records** in one file. STDF files can be multiple GB.

---

## Production-Scale Expansion Tasks

1. **Fix PRR/MIR/WIR parsing** — 1-2 hours, fixes data correctness
2. **Add FAR endianness** — 30 minutes, fixes cross-vendor compatibility
3. **Add HBR/SBR** — 1 hour, enables bin name mapping
4. **Add WCR** — 1 hour, enables accurate wafer map geometry
5. **Add PTR** — 2-3 hours, unlocks parametric analysis (the biggest win)
6. **Add TSR** — 1 hour, enables per-test yield analytics
7. **Add FTR** — 2 hours, enables functional test analysis
8. **Port full generic parser from P5** — 4-6 hours, gets all 26+ record types

For a production semiconductor data pipeline, option 8 (porting the P5 generic parser) is the recommended path — it handles every edge case already.
