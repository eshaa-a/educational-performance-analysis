# curriculum.py
import pandas as pd

class CurriculumOptimizer:
    """
    Very readable mapping from skill name -> recommended modules.
    Also supports simple cost-based ranking if a syllabus with 'est_cost' is provided.
    """
    def __init__(self):
        # curated modules for each skill (presentation-friendly names)
        self.module_map = {
            "python": ["Python Foundations", "OOP & Modules", "Project: Python App"],
            "java": ["Java Basics & OOP", "Collections", "Spring Boot Intro"],
            "c++": ["C++ Core", "STL & Memory", "Competitive C++ Patterns"],
            "javascript": ["JS Essentials", "ES6+, Async JS", "Frontend Interactivity"],
            "html": ["HTML5 Semantics", "Accessibility"],
            "css": ["CSS Layouts", "Responsive Design"],
            "data_structures": ["Arrays & Lists", "Trees & Graphs", "Hashing"],
            "machine_learning": ["Supervised Learning", "Model Evaluation", "Feature Engineering"],
            "deep_learning": ["Neural Networks", "CNNs for Vision", "RNNs & Transformers Intro"],
            "sql": ["SQL Queries & Joins", "Query Optimization"],
            "mongodb": ["NoSQL Modeling", "Aggregation Framework"],
            "data_visualization": ["Matplotlib & Seaborn", "Interactive Dashboards"],
            "operating_systems": ["Processes & Threads", "Memory Management"],
            "computer_networks": ["TCP/IP Basics", "Network Layers & Routing"],
            "dbms": ["ER Modeling", "Transactions & Indexing"],
            "cloud_computing": ["Cloud Concepts", "AWS Basics", "Deploying Services"],
            "cybersecurity": ["Secure Coding Basics", "Authentication & Encryption"],
            "software_engineering": ["Design Patterns", "Testing & TDD"],
            "devops": ["CI/CD Concepts", "Infrastructure as Code"],
            "ai_ml_engineering": ["Model Deployment", "ML Ops Basics"],
            "web_development": ["Fullstack Project", "REST APIs"],
            "mobile_development": ["Android/iOS Basics", "Responsive UI"],
            "git_github": ["Git Workflow", "Pull Requests"],
            "docker": ["Containers Basics", "Docker Compose"],
            "kubernetes": ["K8s Fundamentals", "Deployments & Services"],
            "system_design": ["Scalable Systems", "Design Tradeoffs"]
        }

    def recommend(self, gap_df: pd.DataFrame, top_k=6, syllabus=None, budget=None):
        """
        Produce recommendations:
         - gap_df: DataFrame returned by SkillGapModel.compute_skill_gaps()
         - top_k: how many top skills to recommend modules for
         - syllabus: optional DataFrame with columns ['topic','est_cost'] to consider cost
         - budget: if provided, prune suggestions within budget (greedy)
        Returns a dict: {skill: [modules]}
        """
        skills = gap_df.sort_values("gap", ascending=False).head(top_k).index.tolist()
        rec = {}
        for s in skills:
            if s in self.module_map:
                rec[s] = self.module_map[s]
            else:
                rec[s] = ["Custom training: consult subject expert"]

        # if budget and syllabus provided — do greedy selection: choose cheapest impactful topics
        if budget and syllabus is not None:
            # build cost map
            cost_map = {row.topic: row.est_cost for _, row in syllabus.iterrows()}
            selected = []
            remaining = budget
            for s in skills:
                c = cost_map.get(s, 5)  # default cost
                if c <= remaining:
                    selected.append(s)
                    remaining -= c
            rec = {k: rec[k] for k in selected}
        return rec
