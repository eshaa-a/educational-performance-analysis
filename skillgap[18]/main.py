# main.py
from model import SkillGapModel
from curriculum import CurriculumOptimizer
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    model = SkillGapModel(n_components=2, n_clusters=4)
    model.load_dataframe(df)
    model.preprocess()
    model.run_pca()
    model.run_kmeans()
    gap_df = model.compute_skill_gaps()
    optimizer = CurriculumOptimizer()
    rec = optimizer.recommend(gap_df, top_k=8)
    print("Top skill gaps:")
    print(gap_df.head(10))
    print("\nRecommendations:")
    for k,v in rec.items():
        print(k, "->", v)
