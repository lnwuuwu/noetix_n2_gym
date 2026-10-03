#!/usr/bin/env python
"""读取 TensorBoard 事件文件里的标量，不用起网页。
用法：
  python tbpeek.py <日志目录>                    # 列出所有可用标量名
  python tbpeek.py <日志目录> tag1,tag2,...      # 打印这些标量随 iter 的变化
  python tbpeek.py <日志目录> tag1,tag2 -n 15    # 控制打印行数
目录可以用通配符，会自动取最新的一个。
"""
import sys, glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = sys.argv[1]
if any(c in path for c in '*?'):
    path = sorted(glob.glob(path))[-1]          # 取最新
ea = EventAccumulator(path, size_guidance={'scalars': 0})   # 0 = 全部读，不下采样
ea.Reload()
avail = ea.Tags()['scalars']

if len(sys.argv) < 3:
    print("日志目录:", path)
    print("可用标量 (%d 个):" % len(avail))
    for t in sorted(avail):
        print("   ", t)
    raise SystemExit

tags = [t for t in sys.argv[2].split(',') if t in avail]
missing = [t for t in sys.argv[2].split(',') if t not in avail]
if missing:
    print("!! 不存在的标量:", missing)
nrow = int(sys.argv[sys.argv.index('-n') + 1]) if '-n' in sys.argv else 12

data = {t: {s.step: s.value for s in ea.Scalars(t)} for t in tags}
steps = sorted(data[tags[0]])
print("日志: %s   共 %d 个记录点，最新 iter=%d" % (os.path.basename(path), len(steps), steps[-1]))
print("%7s" % "iter" + ''.join("%16s" % t.split('/')[-1][:15] for t in tags))
for s in steps[::max(1, len(steps) // nrow)] + [steps[-1]]:
    print("%7d" % s + ''.join("%16.4f" % data[t][s] if s in data[t] else "%16s" % "-" for t in tags))
