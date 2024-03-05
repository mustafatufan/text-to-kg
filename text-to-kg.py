import requests
import wikipedia
import spacy
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


# Function to fetch text from Wikipedia
def fetch_wikipedia_text(page_title):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={page_title}&explaintext"
        response = requests.get(url)
        data = response.json()
        page_id = list(data["query"]["pages"].keys())[0]
        return data["query"]["pages"][page_id]["extract"]
    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to extract entities from text
def text_to_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = defaultdict(list)
    for entity in doc.ents:
        if not entity.label_ == "CARDINAL":  # Exclude numerical entities
            entities[entity.label_].append(entity.text)
    return entities


# Function to extract relationships between entities
def entities_to_relationships(entities, text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                obj = [child.text for child in token.head.children if child.dep_ == "dobj"]
                if obj:
                    relationships.append((subject, verb, obj[0]))
    return relationships


# Function to construct a knowledge graph from relationships
def relationships_to_knowledge_graph(relationships):
    knowledge_graph = defaultdict(list)
    for relationship in relationships:
        subject, predicate, object_ = relationship
        knowledge_graph[subject].append((predicate, object_))
        knowledge_graph[object_].append((predicate, subject))  # Add a bidirectional relationship
    return knowledge_graph


def visualize_knowledge_graph(knowledge_graph, output_file, max_nodes=1000, max_edges=10000):
    G = nx.DiGraph()

    # Track node labels and their corresponding entities
    node_labels = defaultdict(list)
    for entity, relations in knowledge_graph.items():
        for relation in relations:
            _, object_ = relation
            node_labels[object_].append(entity)

    # Add nodes and merge duplicate nodes
    for node, entities in node_labels.items():
        G.add_node(node)
        for entity in entities:
            G.add_edge(entity, node)  # Connect the node to its corresponding entities

    # Limit number of nodes and edges
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:max_nodes]
    G = G.subgraph(nodes)
    edges = list(G.edges())[:max_edges]

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust figure size as needed
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=10, node_color="skyblue", font_size=8, width=0.3)

    # Populate edge labels
    edge_labels = {(source, target): label for source, target, label in G.edges(data="label")}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Save the plot to a file with higher resolution
    plt.savefig(output_file, dpi=1200)


# Main function
def main():
    # Fetch text from Wikipedia
    page_title = "Beyoncé"
    text = fetch_wikipedia_text(page_title)
    if text:
        # Extract entities
        entities = text_to_entities(text)

        # Extract relationships
        relationships = entities_to_relationships(entities, text)

        # Construct knowledge graph
        knowledge_graph = relationships_to_knowledge_graph(relationships)
        print("\nKnowledge Graph:")
        for entity, relations in knowledge_graph.items():
            print(entity)
            for relation in relations:
                print(f"  -> {relation[0]}: {relation[1]}")

        visualize_knowledge_graph(knowledge_graph, "knowledge_graph.png")


if __name__ == "__main__":
    main()
