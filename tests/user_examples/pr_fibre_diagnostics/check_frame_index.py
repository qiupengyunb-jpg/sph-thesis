"""Was the earlier 'T~42' label of scan_gap.py actually T=62?"""
import xml.etree.ElementTree as ET
from pathlib import Path

d = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914\r24_broadband_T80"
         r"\output_recon_r24_T80")
fs = sorted(d.glob("LiquidFilmHalf_*.vtp"))          # exactly what scan_gap.py did
print("plain sorted() order used by the earlier probe scripts:")
for idx, label in ((0, "first"), (10, "idx10 'T~14'"), (20, "idx20 'T~28'"),
                   (30, "idx30 'T~42'"), (40, "idx40 'T~80'")):
    p = fs[idx]
    t = float(ET.parse(p).getroot()
              .find('.//FieldData/DataArray[@Name="TimeValue"]').text)
    print("  idx=%2d  %-14s -> %-34s  solver_time=%7.3f  T=%6.2f"
          % (idx, label, p.name, t, t / 0.67082039325))
