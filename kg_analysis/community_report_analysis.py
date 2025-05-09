import pandas as pd

# Read the Parquet file
df = pd.read_parquet("../output/community_reports.parquet")

print('Number of community reports:', len(df))

# For each of the first 10 community reports, print the title, summary, full_content, rank, rating_explanation, and findings.
for i in range(10):
    report = df.iloc[i]
    print(f"Title: {report['title']}")
    print(f"Summary: {report['summary']}")
    print(f"Full Content: {report['full_content']}")
    print(f"Rank: {report['rank']}")
    print(f"Rating Explanation: {report['rating_explanation']}")
    print(f"Findings: {report['findings']}")
    print("~\n"*3)