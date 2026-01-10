from src.core import Timeline
import inspect
print(f"Timeline source: {inspect.getfile(Timeline)}")
print(f"detect_gaps signature: {inspect.signature(Timeline.detect_gaps)}")
