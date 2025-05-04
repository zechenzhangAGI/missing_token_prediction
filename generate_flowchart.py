import json
import argparse
import statistics
import math
from pathlib import Path
import html # To escape potential HTML in text fields

def process_node(node_data, node_list, current_depth, max_depth_info, is_root=False):
    """Recursively processes nodes to calculate scores and structure data for JS."""
    node_id = len(node_list) # Simple unique ID
    node_list.append(node_data) # Add node for summary stats calculation later

    avg_score = None
    scores = node_data.get("judgeScores", [])
    feedback = node_data.get("judgeFeedback", [])
    model_answers = node_data.get("modelAnswers", [])

    # Calculate average score if scores exist for *any* node
    if scores:
        try:
            numeric_scores = [s for s in scores if isinstance(s, (int, float))]
            if numeric_scores:
                avg_score = statistics.mean(numeric_scores)
        except statistics.StatisticsError:
            avg_score = None

    # Update max depth
    max_depth_info['max'] = max(max_depth_info['max'], current_depth)

    # Escape potentially problematic HTML characters in text content for all fields
    query_html = html.escape(node_data.get("problemStatement", "N/A"))
    # Always include problemSolution if it exists
    solution_html = html.escape(node_data.get("problemSolution", "N/A"))
    answers_html = [html.escape(ans) for ans in model_answers]
    feedback_html = [html.escape(fb) for fb in feedback]
    link_desc_html = html.escape(node_data.get("linkDescription", ""))

    # Prepare node structure for JavaScript
    js_node = {
        "name": node_data.get("name", "Unnamed Node"),
        "linkDescription": link_desc_html,
        "query": query_html,
        "solution": solution_html, # Ground truth solution
        "modelAnswers": answers_html, # List of model answers
        "judgeScores": scores, # List of scores for model answers
        "judgeFeedback": feedback_html, # List of feedback for model answers
        "averageScore": avg_score, # Average score for this node (if scores exist)
        # "isLeaf": is_leaf, # No longer needed to determine data presence
        "children": []
    }

    if "children" in node_data and node_data["children"]:
        for child_data in node_data["children"]:
            child_node = process_node(child_data, node_list, current_depth + 1, max_depth_info)
            js_node["children"].append(child_node)
    else:
        js_node["children"] = []

    if is_root:
       js_node['max_depth'] = max_depth_info['max']

    return js_node

def calculate_summary_stats(all_nodes):
    """Calculates summary statistics from the list of all processed nodes."""
    total_nodes = len(all_nodes)
    nodes_with_scores_count = 0
    all_scores = []

    # Collect scores from ALL nodes that have them
    for node in all_nodes:
        scores = node.get("judgeScores", [])
        if scores:
            nodes_with_scores_count += 1
            numeric_scores = [s for s in scores if isinstance(s, (int, float))]
            all_scores.extend(numeric_scores)

    overall_avg_score = None
    if all_scores:
        try:
            overall_avg_score = statistics.mean(all_scores)
        except statistics.StatisticsError:
            overall_avg_score = None

    return {
        "total_nodes": total_nodes,
        "nodes_with_scores": nodes_with_scores_count,
        "overall_avg_score": overall_avg_score # Still on 0-10 scale
    }

def generate_html(tree_data_json, summary_stats, output_path):
    """Generates the HTML file content with embedded data and D3.js logic."""

    # Format summary statistics for display
    score_display = f"{summary_stats['overall_avg_score']:.2f}" if summary_stats['overall_avg_score'] is not None else 'N/A'
    stats_html = f"""
        <p><strong>Total Nodes:</strong> {summary_stats['total_nodes']}</p>
        <p><strong>Nodes with Scores:</strong> {summary_stats['nodes_with_scores']}</p> 
        <p><strong>Max Depth:</strong> {summary_stats['max_depth']}</p>
        <p><strong>Overall Average Score (0-10 scale):</strong> {score_display}</p>
    """ # Adjusted summary text

    # HTML and JS template remains largely the same, but JS logic inside needs update
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interactive Result Flow Chart</title>

  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; }}
    h2 {{ margin: 15px; }}
    .summary-stats {{
        margin: 15px;
        padding: 10px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
    }}
    #chartArea {{
      display: flex;
      flex-direction: row;
      width: 100%;
      height: calc(100vh - 150px); /* Adjusted height for summary box */
    }}
    #chart {{ flex: 3; overflow: auto; position: relative; border-right: 1px solid #ccc; }}
    #infoPanel {{
      flex: 2;
      margin-left: 10px;
      margin-right: 10px;
      overflow-y: auto;
      padding-left: 10px;
    }}
    #infoPanel h3 {{ margin-top: 0; }}
    .detailBox {{ border: 1px solid #ccc; padding: 10px; margin-top: 10px; background: #fdfdfd; }}
    .answer-block {{ border: 1px dashed #eee; padding: 8px; margin-bottom: 8px; background: #fafafa; }}
    .score-list {{ list-style-type: none; padding-left: 0; }}
    .score-list li {{ display: inline-block; margin-right: 5px; background: #eee; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
    .feedback-text {{ font-style: italic; color: #555; font-size: 0.95em; margin-top: 4px;}}
    .section-title {{ font-weight: bold; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #eee; padding-bottom: 3px; }}

    /* Tree styles */
    .node circle {{ stroke-width: 2px; }}
    .node text {{ font: 12px sans-serif; fill: #333; cursor: pointer; }} /* Make text clickable */
    .link {{ fill: none; stroke: #ccc; stroke-width: 2px; cursor: pointer; }}
    .link:hover {{ stroke: #999; }}
  </style>

  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>

<body>
  <h2>Interactive Result Flow Chart</h2>
  <div class="summary-stats">
      <h3>Summary Statistics</h3>
      {stats_html}
  </div>
  <div id="chartArea">
    <div id="chart"></div>
    <div id="infoPanel">
      <h3>Details</h3>
      <div id="details" class="detailBox">
        Click a node or an edge to see details here.
      </div>
    </div>
  </div>

  <script>
    const treeData = {tree_data_json};

    /************************************************************
     * D3 Setup and Rendering Code
     ************************************************************/
    const margin = {{ top: 20, right: 120, bottom: 30, left: 120 }},
          chartWidth  = Math.max(1200, window.innerWidth * 0.55),
          chartHeight = Math.max(800, (treeData.max_depth || 5) * 150);

    const svg = d3.select("#chart")
      .append("svg")
      .attr("width",  chartWidth  + margin.left + margin.right)
      .attr("height", chartHeight + margin.top  + margin.bottom)
      .call(d3.zoom().on("zoom", (event) => {{
         svgGroup.attr("transform", event.transform);
      }}))
      .append("g")
      .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

    const svgGroup = svg;

    const treemap = d3.tree().size([chartHeight, chartWidth]);

    let root = d3.hierarchy(treeData, d => d.children);
    root.x0 = chartHeight / 2;
    root.y0 = 0;

    const colorScale = d3.scaleLinear()
      .domain([0, 5, 10])
      .range(["#d7191c", "#ffffbf", "#1a9641"])
      .clamp(true);

    function collapse(d) {{
      if (d.children && d.children.length > 0) {{
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
      }}
    }}

    // Start fully expanded
    update(root);

    // Update function
    function update(source) {{
      const duration = 750;
      const treeLayout = treemap(root);
      const nodes = treeLayout.descendants();
      const links = treeLayout.descendants().slice(1);

      nodes.forEach(d => {{ d.y = d.depth * 300 }});

      // ---------- NODES ----------
      let node = svgGroup.selectAll("g.node")
        .data(nodes, d => d.id || (d.id = d.data.name + Math.random()));

      let nodeEnter = node.enter().append("g")
        .attr("class", "node")
        .attr("transform", d => `translate(${{source.y0}},${{source.x0}})`)
        .on("click", (event, d) => {{
           if (event.target.tagName === 'circle' || event.target.tagName === 'text') {{
                clickNode(d);
            }}
        }});

      nodeEnter.append("circle")
        .attr("class", "nodeCircle")
        .attr("r", 1e-6)
        .style("stroke", "steelblue")
        .style("fill", d => d._children ? "lightsteelblue" : "#fff");

      nodeEnter.append("text")
        .attr("dy", ".35em")
        .attr("x", d => d.children || d._children ? -13 : 13)
        .attr("text-anchor", d => d.children || d._children ? "end" : "start")
        .text(d => d.data.name)
        .style("fill-opacity", 1e-6);

      let nodeUpdate = nodeEnter.merge(node);

      nodeUpdate.transition()
        .duration(duration)
        .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

      // --- Node Styling Update --- 
      nodeUpdate.select("circle.nodeCircle")
        .attr("r", 8)
        // Stroke color depends only on average score if it exists
        .style("stroke", d => (d.data.averageScore !== null) ? colorScale(d.data.averageScore) : "steelblue")
        .style("stroke-width", "3px")
        // Fill color depends on collapsed state OR average score
        .style("fill", d => {{
            if (d._children) {{ // Collapsed node indicator
                 return "lightsteelblue";
            }} else if (d.data.averageScore !== null) {{ // Color by score if available
                 return colorScale(d.data.averageScore);
            }} else {{ // Default fill if no score and not collapsed
                 return "#fff";
            }}
        }})
        .style("cursor", "pointer");

      nodeUpdate.select("text")
        .style("fill-opacity", 1);

      let nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", d => `translate(${{source.y}},${{source.x}})`)
        .remove();
      nodeExit.select("circle").attr("r", 1e-6);
      nodeExit.select("text").style("fill-opacity", 1e-6);

      // ---------- LINKS ----------
      let link = svgGroup.selectAll("path.link")
        .data(links, d => d.id || (d.id = d.parent.id + "->" + d.data.name));

      let linkEnter = link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", d => {{
          let o = {{ x: source.x0, y: source.y0 }};
          return diagonal(o, o);
        }})
        .on("click", (event, d) => clickEdge(d));

      let linkUpdate = linkEnter.merge(link);
      linkUpdate.transition()
        .duration(duration)
        .attr("d", d => diagonal(d, d.parent));

      let linkExit = link.exit().transition()
        .duration(duration)
        .attr("d", d => {{
          let o = {{ x: source.x, y: source.y }};
          return diagonal(o, o);
        }})
        .remove();

      nodes.forEach(d => {{
        d.x0 = d.x;
        d.y0 = d.y;
      }});
    }}

    function diagonal(s, d) {{
      return `M ${{s.y}} ${{s.x}}
              C ${{ (s.y + d.y) / 2 }} ${{s.x}},
                ${{ (s.y + d.y) / 2 }} ${{d.x}},
                ${{d.y}} ${{d.x}}`;
    }}

    function clickNode(d) {{
      displayNodeDetails(d); // Update info panel first

      // Expand/collapse logic
      if (d.children) {{
          d._children = d.children;
          d.children = null;
      }} else {{
          d.children = d._children;
          d._children = null;
      }}
      if (d.children || d._children) {{
        update(d);
      }}
    }}

    function clickEdge(d) {{
        displayEdgeDetails(d);
    }}

    // --- Updated Node Detail Display --- 
    function displayNodeDetails(d) {{
        let html = `<div class="section-title">Node: ${{d.data.name}}</div>`;
        
        // Always show Problem Statement
        html += `<div class="section-title">Problem Statement</div>`;
        html += `<div>${{d.data.query || "N/A"}}</div>`;
        
        // Always show Ground Truth Solution if it exists and is not 'N/A'
        if (d.data.solution && d.data.solution !== 'N/A') {{
            html += `<div class="section-title">Problem Solution (Ground Truth)</div>`;
            html += `<div style="white-space: pre-wrap; word-wrap: break-word;">${{d.data.solution}}</div>`;
        }}

        // Conditionally show Model Answers section if data exists
        if (d.data.modelAnswers && d.data.modelAnswers.length > 0) {{
            html += `<div class="section-title">Model Answers & Judgements</div>`;
            
            // Display average score if available
            let avgScoreHtml = "N/A";
            if (d.data.averageScore !== null) {{
                const scoreColor = colorScale(d.data.averageScore);
                avgScoreHtml = `<strong style="color:${{scoreColor}}">${{d.data.averageScore.toFixed(2)}} / 10</strong>`;
            }}
            html += `<p><strong>Average Score:</strong> ${{avgScoreHtml}}</p>`;

            // Display individual model answers, scores, and feedback
            d.data.modelAnswers.forEach((answer, index) => {{
                const score = d.data.judgeScores[index] !== undefined ? d.data.judgeScores[index] : 'N/A';
                const scoreDisplay = typeof score === 'number' ? score.toFixed(1) : score;
                const feedback = d.data.judgeFeedback[index] || 'No feedback provided.';
                
                html += `<div class="answer-block">`;
                html += `<p><strong>Answer ${{(index + 1)}} (Score: ${{scoreDisplay}} / 10):</strong></p>`;
                html += `<div style="white-space: pre-wrap; word-wrap: break-word;">${{answer}}</div>`; // Preserve formatting
                html += `<p class="feedback-text"><em>Feedback:</em> ${{feedback}}</p>`;
                html += `</div>`;
            }});
        }} else {{
             // Optionally indicate if no model answers were generated for this node
             // html += `<p><em>No model answers were generated for this node.</em></p>`;
        }}

        d3.select("#details").html(html);
        retypesetMath(); // Retypeset MathJax
    }}

    function displayEdgeDetails(d) {{
        let parentName = d.parent ? d.parent.data.name : "Root";
        let childName = d.data.name;
        let linkDesc = d.data.linkDescription || "No link description provided.";
        let html = `
            <strong>Edge:</strong> ${{parentName}} &rarr; ${{childName}}<br/><br/>
            <strong>Logical Connection / Link Description:</strong>
            <p>${{linkDesc}}</p>
        `;
        d3.select("#details").html(html);
        retypesetMath();
    }}

    function retypesetMath() {{
         if (window.MathJax && window.MathJax.typesetPromise) {{
            Promise.resolve(window.MathJax.typesetPromise([document.getElementById('details')])).catch((err) => console.error('MathJax typesetting failed:', err));
         }} else if (window.MathJax && window.MathJax.typeset) {{
             window.MathJax.typeset();
         }}
    }}

  </script>
</body>
</html>
"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully generated HTML flowchart: {output_path}")
    except Exception as e:
        print(f"Error writing HTML file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate an interactive D3 flowchart from specific JSON result data.")
    parser.add_argument("input_json", help="Path to the input JSON file containing 'config' and 'tree' keys.")
    parser.add_argument("-o", "--output", help="Path to the output HTML file (defaults to input filename with .html extension).")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}")
        return

    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return
    except Exception as e:
        print(f"Error opening input file: {e}")
        return

    if "tree" not in data:
        print("Error: JSON data must contain a 'tree' key at the top level.")
        return

    tree_root_data = data["tree"]

    all_nodes_list = []
    max_depth_info = {'max': 0}

    processed_tree_data = process_node(tree_root_data, all_nodes_list, 0, max_depth_info, is_root=True)

    summary_stats = calculate_summary_stats(all_nodes_list)
    summary_stats['max_depth'] = max_depth_info['max']

    tree_data_json_string = json.dumps(processed_tree_data, indent=None, ensure_ascii=False)

    generate_html(tree_data_json_string, summary_stats, output_path)

if __name__ == "__main__":
    main() 