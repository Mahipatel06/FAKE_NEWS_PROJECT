import matplotlib.pyplot as plt

def plot_label_distribution(fake_count, real_count):
    fig, ax = plt.subplots()
    ax.bar(['Fake', 'Real'], [fake_count, real_count], color=['red', 'green'])
    ax.set_ylabel("Count")
    ax.set_title("Fake vs Real News Count")
    return fig

def plot_article_length_histogram(df):
    lengths = df['text'].str.len()
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel("Article Length (Characters)")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Distribution of News Article Lengths")
    return fig

def plot_label_pie_chart(fake_count, real_count):
    labels = ['Fake', 'Real']
    sizes = [fake_count, real_count]
    colors = ['red', 'green']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')  # Equal aspect ratio makes pie chart circular
    ax.set_title("Fake vs Real News Percentage")
    return fig
