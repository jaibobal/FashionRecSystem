# VogueVision
Crowd-sourced fashion recommendation system built using ResNet50 trained on Myntra database. Python libraries used include tensorflow, keras, streamlit, pickle, numpy, os, SciKitLearn and PIL.

Steps to use:
1. Create new folder 'ABC'.
2. Move app.py and main.py to 'ABC', along with embeddings.pkl and filenames.pkl.
3. Within 'ABC', create an empty folder 'temp'
4. Download training database from: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
5. Store all downloaded images in a folder named 'data', and move this folder into 'ABC'
6. Run main.py through terminal using "$ streamlit run main.py" This should take you to your localhost website, where the recommendation model should work as expected.
