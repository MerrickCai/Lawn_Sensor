// import d3 library
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

export default function createChart(classCounts) {
  // ---------------- Build D3.js Bar Chart (Species Distribution) ----------
  const chartData = Object.entries(classCounts).map(([cls, count]) => ({
    class: cls,
    count,
  }));

  const margin = { top: 40, right: 20, bottom: 60, left: 60 },
    width = 800 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom;

  const svgBar = d3
    .select("#bar-chart")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3
    .scaleBand()
    .domain(chartData.map((d) => d.class))
    .range([0, width])
    .padding(0.2);

  const y = d3
    .scaleLinear()
    .domain([0, d3.max(chartData, (d) => d.count)])
    .nice()
    .range([height, 0]);

  svgBar
    .append("g")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x))
    .selectAll("text")
    .attr("transform", "rotate(-45)")
    .style("text-anchor", "end");

  svgBar.append("g").call(d3.axisLeft(y));

  // Axis Labels
  svgBar
    .append("text")
    .attr("x", width / 2)
    .attr("y", height + margin.bottom - 10)
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .text("Plant Species");

  svgBar
    .append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("y", -margin.left + 15)
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .text("Count");

  svgBar
    .selectAll(".bar")
    .data(chartData)
    .enter()
    .append("rect")
    .attr("class", "bar")
    .attr("x", (d) => x(d.class))
    .attr("y", (d) => y(d.count))
    .attr("width", x.bandwidth())
    .attr("height", (d) => height - y(d.count))
    .attr("fill", "#4CAF50");

  // Data labels on bars
  svgBar
    .selectAll(".label")
    .data(chartData)
    .enter()
    .append("text")
    .attr("class", "label")
    .attr("x", (d) => x(d.class) + x.bandwidth() / 2)
    .attr("y", (d) => y(d.count) - 5)
    .attr("text-anchor", "middle")
    .style("font-size", "12px")
    .text((d) => d.count);

  // ------------- Build D3.js Pie Chart (Species Distribution) --------------
  const pieWidth = 400,
    pieHeight = 400;
  const radius = Math.min(pieWidth, pieHeight) / 2;

  const svgPie = d3
    .select("#pie-chart")
    .append("svg")
    .attr("width", pieWidth)
    .attr("height", pieHeight)
    .append("g")
    .attr("transform", `translate(${pieWidth / 2},${pieHeight / 2})`);

  const pie = d3
    .pie()
    .value((d) => d.count)
    .sort(null);

  const arc = d3.arc().innerRadius(0).outerRadius(radius);

  const color = d3
    .scaleOrdinal()
    .domain(chartData.map((d) => d.class))
    .range(d3.schemeCategory10);

  const pieData = pie(chartData);

  const arcs = svgPie
    .selectAll("arc")
    .data(pieData)
    .enter()
    .append("g")
    .attr("class", "arc");

  arcs
    .append("path")
    .attr("d", arc)
    .attr("fill", (d) => color(d.data.class))
    .attr("stroke", "#fff")
    .attr("stroke-width", "2px");

  arcs
    .append("text")
    .attr("transform", (d) => `translate(${arc.centroid(d)})`)
    .attr("text-anchor", "middle")
    .attr("font-size", "12px")
    .attr("fill", "#fff")
    .text((d) => d.data.class);
}
