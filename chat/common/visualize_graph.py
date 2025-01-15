from chat.main_graph.graph import graph as main_graph
from chat.supervisor_graph.graph import graph as supervisor_graph
from chat.api_request_checker_graph.graph import graph as api_request_checker_graph

#for supervisor graph
def visualize_supervisor_graph():
    try:
        graph_viz = supervisor_graph.get_graph().draw_mermaid_png()
        with open("supervisor_graph_visualization.png", "wb") as f:
            f.write(graph_viz)
        print("Graph visualization saved as 'supervisor_graph_visualization.png'")
    except Exception as e:
        print("Could not generate graph visualization:", str(e))


    
#for main graph
def visualize_main_graph():
    try:
        graph_viz = main_graph.get_graph().draw_mermaid_png()
        with open("main_graph_visualization.png", "wb") as f:
            f.write(graph_viz)
        print("Graph visualization saved as 'main_graph_visualization.png'")
    except Exception as e:
        print("Could not generate graph visualization:", str(e))


#for api request checker graph
def visualize_api_request_checker_graph():
    try:
        graph_viz = api_request_checker_graph.get_graph().draw_mermaid_png()
        with open("api_request_checker_graph_visualization.png", "wb") as f:
            f.write(graph_viz)
        print("Graph visualization saved as 'api_request_checker_graph_visualization.png'")
    except Exception as e:
        print("Could not generate graph visualization:", str(e))

if __name__ == "__main__":
    visualize_supervisor_graph() 
    # visualize_main_graph()
    # visualize_api_request_checker_graph()


