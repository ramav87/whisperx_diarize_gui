import sys
import os

# Add the 'src' folder to the system path so Python finds your package
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Now import the main function from your package properly
from diarize_gui.gui import main

if __name__ == '__main__':
    main()