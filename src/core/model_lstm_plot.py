import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Show HFACS Taxonomy distribution
def show_label_distribution(data, ds_name='asrs'):
    plt.figure(figsize=(8,6))
    sns.countplot(data['HFACS_Category_Value_Predict'])
    plt.title(f'The distribution of HFACS Taxonomy (DataSet={ds_name})')

# Show Text frequent of word count distribution
def word_count_distribution(df, ds_name='asrs'):
    all_words = [word for text in df['narrative'].values for word in text.split()]
    word_counts = Counter(all_words)

    # Plot top 50 most frequent words
    common_words = word_counts.most_common(50)
    labels, values = zip(*common_words)
    plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.title(f'The distribution of frequent of Word Count (DataSet={ds_name})')
    plt.show()

    print(f"Total unique words: {len(word_counts)}")

# Show Text word count distribution
def show_narrative_distribution(data, ds_name='asrs'):
    plt.figure(figsize=(14, 6))

    word_count = data['narrative_word_count']

    sns.histplot(word_count,  bins=50, color='blue', kde=True)
    plt.xlabel('Number of Words')
    plt.ylabel('Number of sample')
    plt.title(f'Distribution of Word Counts (DataSet={ds_name})')

    mean = word_count.mean()
    std = word_count.std()

    # Display mean and standard deviation on the plot
    # code adapted from (Controlling Style of Text and Labels Using a Dictionary — Matplotlib 3.9.2 Documentation, n.d.)
    plt.text(0.7, 0.9, r'$\mu={:.2f}$'.format(mean), transform=plt.gca().transAxes, fontdict={'size': 20})
    plt.text(0.7, 0.85, r'$\sigma={:.2f}$'.format(std), transform=plt.gca().transAxes, fontdict={'size': 20})
    # code adapted end

    # Display the plots
    plt.tight_layout()
    plt.show()

#  Show Loss and Accuracy
def model_summary(history, ds_name='asrs'):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Code adapted from Li (2021)
    # Plot the Loss
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='test')
    axes[0].set_title(f'Loss (DataSet={ds_name})')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot the accuracy
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='test')
    axes[1].set_title(f'Accuracy (DataSet={ds_name})')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    # Code adapted end

    plt.tight_layout()
    plt.show()

#  Show Confusion Matrix
def show_lstm_confusion_matrix(conf_matrix, classes, ds_name='asrs'):
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix LSTM (DataSet={ds_name})')
    plt.show()