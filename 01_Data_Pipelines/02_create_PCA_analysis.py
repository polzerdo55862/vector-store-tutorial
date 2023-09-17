##########################################################################################################
# Create PCA plot using embeddings df
##########################################################################################################
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_pca_plot(embeddings_df):
    '''
    The function performs a principal component analysis to reduce the dimensions to 2 so we can print them in a plot

    Parameters
        - embeddings_df (DataFrame): data frame with the columns "text_chunk" and "embeddings"

    Returns
        - df_reduced (DataFrame): data frame with the 2 most relevant Principal Components
    '''
    # Perform PCA with 2 components
    pca = PCA(n_components=2)

    # apply principal component analysis to the embeddings table
    df_reduced = pca.fit_transform(embeddings_df[embeddings_df.columns[:-2]])

    # Create a new DataFrame with reduced dimensions
    df_reduced = pd.DataFrame(df_reduced, columns=['PC1', 'PC2'])

    ############################################################################################
    # Create a scatter plot
    ############################################################################################
    def create_scatter_plot(df_reduced):
        plt.scatter(df_reduced['PC1'], df_reduced['PC2'], label=df_reduced['PC2'])

        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter Plot')

        # Add labels to each dot
        for i, label in enumerate(embeddings_df.iloc[: , -1].to_list()):
            plt.text(df_reduced['PC1'][i], df_reduced['PC2'][i], label)

        # Save and display the plot
        plt.savefig('../02_Data/principal_component_plot.png', format='png')

    # create and save scatter plot
    create_scatter_plot(df_reduced=df_reduced)

    return df_reduced

# Load embeddings_df.csv into data frame
embeddings_df = pd.read_csv('../02_Data/embeddings_df.csv')

# use the function create_pca_plot to
df_reduced = create_pca_plot(embeddings_df)