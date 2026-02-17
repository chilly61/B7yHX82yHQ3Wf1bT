import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords


df = pd.read_csv("ProjectC.csv")

# -----------------------------
# 2. Text cleaning
# -----------------------------
df['job_title_clean'] = df['job_title'].astype(str).str.strip().str.lower()
df['job_title_clean'] = df['job_title_clean'].str.translate(
    str.maketrans('', '', string.punctuation))


# -----------------------------
# 4. Unigram (single word) analysis
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

all_words = df['job_title_clean'].str.split().sum()
words_filtered = [w for w in all_words if w not in stop_words]
word_counts = Counter(words_filtered)
top_unigrams = word_counts.most_common(10)

# Plot top unigrams
words, counts = zip(*top_unigrams)
plt.figure(figsize=(12, 6))
ax = plt.bar(words, counts, color='orange')
plt.xticks(rotation=45)
plt.xlabel('Word')
plt.ylabel('Count')
plt.title('Top 10 Words in Job Titles')

# 在每个柱子上加数字
for p in ax:
    plt.text(p.get_x() + p.get_width()/2, p.get_height(), str(int(p.get_height())),
             ha='center', va='bottom', fontsize=10)

plt.show()
# -----------------------------
# 5. Bigram (2-word) analysis
# -----------------------------
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
X2 = vectorizer.fit_transform(df['job_title_clean'])
sum_words = X2.sum(axis=0)
words_freq = [(word, sum_words[0, idx])
              for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
top_bigrams = words_freq[:10]

# Plot top bigrams
bigram_words, bigram_counts = zip(*top_bigrams)
plt.figure(figsize=(12, 6))
ax = plt.bar(bigram_words, bigram_counts, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Bigram')
plt.ylabel('Count')
plt.title('Top 10 Bigrams in Job Titles')

# 在每个柱子上加数字
for p in ax:
    plt.text(p.get_x() + p.get_width()/2, p.get_height(), str(int(p.get_height())),
             ha='center', va='bottom', fontsize=10)

plt.show()

'''# EDA for location
US_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
    'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
    'Wisconsin', 'Wyoming'
]

regions = {
    'Northeast': ['New York', 'Massachusetts', 'Pennsylvania', 'New Jersey', 'Connecticut', 'Rhode Island', 'Maine', 'Vermont', 'New Hampshire'],
    'Midwest': ['Illinois', 'Ohio', 'Michigan', 'Indiana', 'Wisconsin', 'Minnesota', 'Iowa', 'Missouri', 'Kansas', 'Nebraska', 'North Dakota', 'South Dakota'],
    'South': ['Texas', 'Florida', 'Georgia', 'North Carolina', 'South Carolina', 'Virginia', 'Tennessee', 'Alabama', 'Louisiana', 'Arkansas', 'Mississippi', 'Oklahoma', 'Kentucky', 'West Virginia', 'Maryland', 'Delaware'],
    'West': ['California', 'Washington', 'Oregon', 'Nevada', 'Arizona', 'Colorado', 'Utah', 'New Mexico', 'Idaho', 'Montana', 'Wyoming', 'Hawaii', 'Alaska', 'Arizona']
}

country_alias = {
    'United States': ['United States', 'USA', 'US', 'Amerika Birleşik Devletleri'],
    'Canada': ['Canada', 'Kanada'],
    'Turkey': ['Türkiye', 'Turkey', 'Turkiye']
}

area_to_state = {
    'San Francisco Bay Area': 'California',
    'Greater Philadelphia Area': 'Pennsylvania',
    'Greater Boston Area': 'Massachusetts',
    'Dallas/Fort Worth Area': 'Texas',
    'Greater Atlanta Area': 'Georgia',
    'Greater Chicago Area': 'Illinois',
    'Greater Los Angeles Area': 'California'
}


def classify_location(loc):
    loc = loc.strip()

    # 先检查是否在已知都市圈映射表
    if loc in area_to_state:
        return 'US Metro/Area'

    # 检查是否包含州名
    is_us = any(state in loc for state in US_states)

    if is_us:
        if 'Area' in loc or 'Greater' in loc or 'Bay' in loc:
            return 'US Metro/Area'
        else:
            return 'US City'
    else:
        # 非美国地区
        for standard_name, aliases in country_alias.items():
            if any(alias in loc for alias in aliases):
                if standard_name == 'United States':
                    return 'US Metro/Area'
                else:
                    return 'Non-US Country'
        return 'Non-US City'


df['location_type'] = df['location'].apply(classify_location)

# 统计每类数量
location_counts = df['location_type'].value_counts()
print(location_counts)

# -----------------
# 1️⃣ 柱状图
plt.figure(figsize=(8, 5))
sns.barplot(x=location_counts.index,
            y=location_counts.values, palette="pastel")
plt.title("Location Type Distribution")
plt.ylabel("Count")
plt.xlabel("Location Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------
# 2️⃣ 饼图
plt.figure(figsize=(6, 6))
plt.pie(location_counts.values, labels=location_counts.index,
        autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Location Type Proportion")
plt.show()


def assign_region(loc):
    loc = loc.strip()

    # 优先使用都市圈映射表
    state = area_to_state.get(loc, None)

    # 如果找不到，再尝试匹配州名
    if not state:
        state = next((s for s in US_states if s in loc), None)

    # 匹配四大区域
    if state:
        for region, states in regions.items():
            if state in states:
                return region
    if loc in ['United States', 'USA', 'US', 'Amerika Birleşik Devletleri']:
        return 'US?'

    return 'Non-US'


# 直接用 location 列统一归类四大区域 + Non-US
df['US_region'] = df['location'].apply(assign_region)

# 绘图
plt.figure(figsize=(8, 5))
sns.barplot(x=df['US_region'].value_counts().index,
            y=df['US_region'].value_counts().values,
            palette="pastel")
plt.title("Region Distribution (Including Non-US)")
plt.ylabel("Count")
plt.xlabel("Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.pie(df['US_region'].value_counts().values,
        labels=df['US_region'].value_counts().index,
        autopct='%1.1f%%',
        colors=sns.color_palette("pastel"))
plt.title("Region Proportion (Including Non-US)")
plt.show()
'''

''' Connection''''''
# Preprocess
df['connection_clean'] = df['connection'].astype(str).str.strip()
df['is_500_plus'] = df['connection_clean'] == '500+'
df['connection_num'] = df['connection_clean'].replace(
    '500+', '500').astype(float)

# Define bins for remaining data
bins = [0, 50, 100, 200, 300, 400, 499]  # 499 以下分箱
labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-499']
df['bin'] = pd.cut(df['connection_num'], bins=bins,
                   labels=labels, include_lowest=True)
df['bin'] = df['bin'].cat.add_categories('500+')
df.loc[df['is_500_plus'], 'bin'] = '500+'

# Count per bin
counts = df['bin'].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 5))
ax = counts.plot(kind='bar', color='skyblue')
plt.xlabel('Connection Range')
plt.ylabel('Count')
plt.title('Connection Distribution with 500+ Highlighted')
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.show()
''''''Connection '''
