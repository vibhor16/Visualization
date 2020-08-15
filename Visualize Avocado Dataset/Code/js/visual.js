 const margin = 60;
 const width = 1000 - 2 * margin;
 const height = 600 - 2 * margin;
 var labels;
 var label_size = '15px';
 var letter_spacing = 1;
 var bandwidth;
 var keyValueMapper;
 var maximumX = 0;
 var maximumY = 0;
 var xaxis_ticks;
 var yaxis_ticks;
 var yScale;
 var xScale;
 var enhanced_xscale;
 var hist_data;
 var valuesX; 
 var valuesX_copy; 
 var valuesY;
 var xAxisGen;
 var axis_transition_time = 1000;

 
 // Renders either bar chart or a histogram
 function renderGraph(feature_name, data, type, nBin){   

    // Initializes the label content for the graphs in a map
    initLabels();


    if(type == "CATEGORICAL"){

        // Hide the slider and show bar graph for Categorical features
        hideSlider();
        renderBarGraph(feature_name, data);
    } else if(type == "NUMERICAL"){

        // Show the slider and show histogram graph for Numerical features
        showSlider();
        renderHistogram(feature_name, data, nBin);
    }

 }

 function initLabels(){
    labels = {'Type': {
                        'x':'Avocado Growth Type',
                        'y':'Avocado Count',
                        'title': 'Counts of Avocado Grown with a Given Type'
                    },
             'Year': {
                        'x':'Avocado Sales Year',
                        'y':'Avocado Sales Count',
                        'title': 'Counts of Avocado Sold (2015 - 2020)'
                    },   
             'City': {
                        'x':'Avocado Sales City',
                        'y':'Avocado Sales Count',
                        'title': 'Count of Avocado Sold in a City (2015 - 2020)'
                    },
             'Price': {
                        'x':'Average Price of Avocado ($)',
                        'y':'Avocado Sales Count',
                        'title': 'Counts of Avocado Sold in a Given Price Range'
                    },
             'Total Volume': {
                        'x':'Avocado Count',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Sold in a Given Count Range'
                    },
             'Total Bags': {
                        'x':'Avocado Bags Bought',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Bags Bought in a Given Range Bag Count '
                    },

              'Rating': {
                        'x':'Avocado Rating',
                        'y':'Rating Frequency',
                        'title': 'Avocado Rating (1 to 10) as a Fruit by Markets'
                    },
             'Major Size': {
                        'x':'Avocado Size',
                        'y':'Avocado Size Count',
                        'title': 'Count of a Given Avocado Size Sold in Markets'
                    },   
             'Population': {
                        'x':'City Population Range',
                        'y':'Frequency',
                        'title': 'Population Range of Avocado Markets\' City'
                    },
             'XLarge Bags': {
                        'x':'Avocado Bags Bought',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Bags (XLarge) Bought in a Given Range Bag Count '
                    },
             'Large Bags': {
                        'x':'Avocado Bags Bought',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado (Large) Bags Bought in a Given Range Bag Count '
                    },
             'PLU4046': {
                        'x':'Avocado Sold',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Size PLU4046 Sold'
                    },
              'PLU4225': {
                        'x':'Avocado Sold',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Size PLU4225 Sold'
                    },
             'PLU4770': {
                        'x':'Avocado Sold',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Size PLU4770 Sold'
                    },
             'Small Bags': {
                        'x':'Avocado Bags Bought',
                        'y':'Frequency',
                        'title': 'Frequency of Avocado Bags (Small) Bought in a Given Range Bag Count '
                    }

            };
 }

 function hideSlider(){
    slider = document.getElementById("theSlider");
    if(slider !=null && slider != undefined)
         slider.style.visibility = "hidden";
 }

  function showSlider(){
    slider = document.getElementById("theSlider");
    if(slider !=null && slider != undefined)
         slider.style.visibility = "visible";
 }


// Called when the slider value changes or the drag functionality is used
 function update_histogram(feature_name, data, nBin){

    var svg = d3.select("svg");
    
    // Reload the fresh data from csv
    load_data(feature_name, data);

    // Update X Axis details
    update_max_x();
    update_x_axis(feature_name, data, nBin , svg); 
    
    // Add transition to X Axis
    svg.select('#x-axis')
    .transition().duration(axis_transition_time)
    .call(xAxisGen);
   
    // Update Y Axis details
    yaxis_ticks = svg.select('#y-axis');
    update_y_axis();

    // Re render the bars of the histogram based upon new bin values
    render_histo_bars(nBin);
 }

 
 function update_max_x(){
     maximumX = d3.max(hist_data, function(d) { return d;} );
 }


 function load_data(feature_name, data){

    // Data is loaded into the hist_data array
    hist_data = [];
    data.map(function(d) {
        hist_data.push(+d[feature_name]);
    })

 }

 function update_y_axis(){

    // Linear scale used to divide the domain on the range on y axis
    yScale = d3.scaleLinear();
    yScale.range([height, 0])
    .domain([0, maximumY + 50]);

    // Y ticks and the y axis are added to the svg
    yaxis_ticks
    .attr('transform', `translate(`+margin+`, 0)`)
    .transition().duration(axis_transition_time)
    .call(d3.axisLeft(yScale));
    

 }

 function update_x_axis(feature_name, data, nBin){
     maximumY = 0;

     // Linear scale used to divide the domain on the range on x axis
     xScale = d3.scaleLinear()
    .range([0, width])
    .domain([0, maximumX]);

    // xMap contains the x locations of the respective x ticks on the svg
    var xMap = axis_scaler(nBin, xScale.range(), xScale.domain());

    // valuesX has the values for the x ticks
    valuesX = Object.keys(xMap);
    
    // frequency_keyValueMappererX has the y axis count for each of the x axis bin range
    var frequency_keyValueMappererX = d3.nest()
                .key(function(d) { return d[feature_name]; })
                .sortKeys(d3.ascending)
                .rollup(function(leaves) { 
                    return leaves.length; 
                })
                .entries(data)

    valuesY = {};
    // Converts to integer based map
    frequency_keyValueMappererX = getValue(frequency_keyValueMappererX);

    for(i = 0;i<valuesX.length;i++){
        count = 0;
        for(j = valuesX[i]; j < parseInt(valuesX[i]) + parseInt(bandwidth); j++){
            if(frequency_keyValueMappererX[parseInt(j)] !=  undefined ){
                count += parseInt(frequency_keyValueMappererX[parseInt(j)]);
            }
        }

        if(j == maximumX){
            if(frequency_keyValueMappererX[parseInt(j)] != undefined){
                count += parseInt(frequency_keyValueMappererX[parseInt(j)]);
            }
        }
        valuesY[valuesX[i]] = count;

        // Finds the max value of frequency to provide it to the y axis
        if(count > maximumY)
            maximumY = count;
    }

    valuesX_copy = valuesX.slice();
    for(i=1;i<=3;i++){
        valuesX_copy.push(parseInt(valuesX_copy[valuesX_copy.length - 1]) + parseInt(bandwidth));
    }
    
    enhanced_xscale = d3.scaleLinear()
    .range([0, width])
    .domain([0, maximumX+ 2 * bandwidth]);
   

    valuesX_copy.pop()
    xAxisGen = d3.axisBottom(enhanced_xscale);
    xAxisGen.ticks(nBin + 1);
    xAxisGen.tickValues(valuesX_copy);
 }
 

 function getValue(frequency_keyValueMappererX){
    var map = {};
    for(i=0;i<Object.keys(frequency_keyValueMappererX).length; i++){
        map[parseInt(frequency_keyValueMappererX[i].key)] = parseInt(frequency_keyValueMappererX[i].value);
    }
    return map;
 }

 // Re renders the bars in histogram based on the bin values
 function render_histo_bars(nBin){


    d3.select('#ref-line').remove();
    d3.select('#ref-text').remove();
    d3.select('.d3-tip').remove();

    var chart = d3.select('svg').select('g');
    keyValueMapper = [];
    for(i=0;i<valuesX.length;i++){
        keyValueMapper[i] = {};
        keyValueMapper[i].key = valuesX[i]; 
        keyValueMapper[i].value = valuesY[valuesX[i]];
    }

    // Color schema for the bars
    var colorSchema = d3.scaleOrdinal()
        .domain(valuesX)
        .range(d3.schemeSet3);

    var rectWidth;
    if(nBin == 1){
        // Width of a bar is maximum X value for nBin = 1
        rectWidth = Math.ceil(parseInt(enhanced_xscale(maximumX)));
    }
     else {
        // Width of a bar is the xScale value for nBin > 1
        rectWidth = Math.ceil(enhanced_xscale(valuesX[1]));
    }

    var x_bar_val = {};
    var nextVal = 0;
    for(i=0;i<valuesX.length;i++){
        x_bar_val[valuesX[i]] = nextVal;
        nextVal += rectWidth;
    }


    // Tip on the bar when hovered upon
    var tip = d3.tip()
      .attr('class', 'd3-tip')
      .offset([-10, 0])
      .html(function(d) {      
          return "<span style='color:"+colorSchema(d.key)+"'> Range - [" + d.key + ", " + (parseInt(d.key) + parseInt(bandwidth)) + ") <br> Frequency - " + d.value + "</span>";
    })
    
    chart.call(tip);


    // Remove the existing bars
    d3.selectAll("rect").remove();

    // Render the bars
    chart.selectAll()
            .data(keyValueMapper)
            .enter()
            .append('rect')
            .attr('x', (s) => enhanced_xscale(s.key)+margin)
            .attr('y', (s) => height)
            .attr('height', 0)
            .attr("opacity", 0.8)
            .attr('width', rectWidth)
            .attr("fill", (s) => colorSchema(s.key))
            .on('mouseover', tip.show)
            .on('mouseout', tip.hide)
            .on('mouseenter', function (s, i) {
                d3.select(this).raise();

                // Increase width and make it higher
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('opacity', 1)
                    .attr('x', (s) => enhanced_xscale(s.key) + margin -5)
                    .attr('y', (s) => yScale(s.value))
                    .attr('width', rectWidth + 10)
                    .attr('height', (s) => height - yScale(s.value) + 10)
                    .style("transform", "scale(1,0.979)"); 

                // Reference line for y values of rect    
                d3.select('svg').select('g')
                .append('line')
                .attr('id','ref-line')
                .attr('x1', 0)
                .attr('y1', yScale(s.value))
                .attr('x2', width)
                .attr('y2', yScale(s.value))
                .attr('transform','translate('+margin+',0)')
                .attr("stroke-width", 1)
                .attr("stroke", "red");

                // Y value for hovered bar on the right
                d3.select('svg').select('g')
                .append('text')
                .attr('id','ref-text')
                .attr('x', width + margin + 5)
                .attr('y', yScale(s.value))
                .style('fill','white')
                .text(s.value);
                
            })
            .on('mouseleave', function (actual, i) {

                // Reset the bar width and height
                d3.select(this)
                .attr("opacity", 0.8)
                .transition()
                .duration(200)
                .attr('x', (s) => enhanced_xscale(s.key) + margin)
                .attr('y', (s) => yScale(s.value))
                .attr('width',rectWidth)
                .attr('height', (s) => height - yScale(s.value))
                .style("transform", "scale(1,1)");

                // Remove ref line
                d3.select('#ref-line').remove();
                d3.select('#ref-text').remove();
            
            })

    // Add transition when rendering the bars
    const t = d3.transition()
      .duration(axis_transition_time);

    chart.selectAll('rect')
    .transition(t)
    .attr('height', (s) => height - yScale(s.value))
    .attr('y', (s) => yScale(s.value));
 }

 function sliderListener(el){

    // Updates the slider value for nBins
     var slider = document.getElementById("theSlider");
     nBin = 21 - el.value;
     update_histogram(FEATURE_NAME, DATA, nBin);
    
 }

 // Get the slider back to defaults
 function resetSlider(bins){
    document.getElementById("theSlider").value = bins;
 }


 function renderHistogram(feature_name, data, nBin){

    // Clear existing graph
    d3.selectAll("svg").remove();
    var vla = 100;
    var whichBtn = -1;
    var oldX;

    var svg = d3.select("#graph_area")
    .append("svg")
    .attr("width", "73.75em")
    .style("border", "1px solid")
    .attr("height", "42em")
    // Mouse events added to implement drag
    .on("mousedown", function(){
        whichBtn = 1;
        oldX = d3.event.pageX;
    })
    .on("mouseup", function() {
        whichBtn = -1;
        document.body.style.cursor = "initial";
    })
    .on("mouseout", function() {
        whichBtn = -1;
        document.body.style.cursor = "initial";
    })
    .on("mousemove", function(){
        // Left click
        if(whichBtn == 1){
            if(d3.event.pageX < oldX){
                // left
                if(nBin < maxSlider){
                    document.body.style.cursor = "w-resize";
                    nBin = nBin + 1;
                    if(nBin != undefined)
                    update_histogram(FEATURE_NAME, DATA, nBin);
                  }
                  
            } else {
                 if(nBin > minSlider){
                    document.body.style.cursor = "e-resize";
                    nBin = nBin - 1;
                    update_histogram(FEATURE_NAME, DATA, nBin);
                  }
                  
            }
            // Update slider value
            document.getElementById("theSlider").value = 21 - nBin;
        } 
    });
    

    var chart = svg.append('g')
    .attr('transform',`translate(${margin}, ${margin})`);
    
    load_data(feature_name, data);
    update_max_x();
   
    update_x_axis(feature_name, data, nBin , svg); 
    
    // Append the x axis
    chart.append('g')
    .attr('id','x-axis')
    .attr('transform', `translate(`+margin+`, ${height})`)
    .transition().duration(axis_transition_time)
    .call(xAxisGen);
    
    yaxis_ticks = chart.append('g').attr('id','y-axis');
    update_y_axis();

    // Color schema for the bars
    var colorSchema = d3.scaleOrdinal()
       .domain(data.map((s) => s.key))
       .range(d3.schemeSet3);

    // Tip added on bar hover
    var tip = d3.tip()
      .attr('class', 'd3-tip')
      .offset([-10, 0])
      .html(function(d) {      
          return "<span style='color:"+colorSchema(d.key)+"'> Range - [" + d.key + ", " + (parseInt(d.key) + parseInt(bandwidth)) + ") <br> Frequency - " + d.value + "</span>";
      })
    
    svg.call(tip);
   

    keyValueMapper = [];
    for(i=0;i<valuesX.length;i++){
        keyValueMapper[i] = {};
        keyValueMapper[i].key = valuesX[i]; 
        keyValueMapper[i].value = valuesY[valuesX[i]];
    }

    var rectWidth;
    var x_bar_val = {};

    if(nBin == 1)
        // maximum x value for bar width when nBin = 1
        rectWidth = Math.ceil(enhanced_xscale(maximumX));
     else 
        // bar width set when nBin > 1
        rectWidth = Math.ceil(enhanced_xscale(valuesX[1]));

   

    var nextVal = 0;
    for(i=0;i<valuesX.length;i++){
        x_bar_val[valuesX[i]] = nextVal;
        nextVal += rectWidth;
    }


    // Render the bars on the svg
    var bars = chart.selectAll()
            .data(keyValueMapper)
            .enter()
            .append('rect')
            .attr('x', (s) => enhanced_xscale(s.key) + margin)
            .attr('y', (s) => height)
            .attr("opacity", 0.8)
            .attr('width', rectWidth)
            .attr('height',  0)
            .attr("fill", (s) => colorSchema(s.key))
            .on('mouseover', tip.show)
            .on('mouseout', tip.hide)
            .on('mouseenter', function (s, i) {
                d3.select(this).raise();

                // Increase bar width and height on mouseenter event
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('opacity', 1)
                    .attr('x', (s) => enhanced_xscale(s.key) + margin -5)
                    .attr('y', (s) => yScale(s.value))
                    .attr('width', rectWidth+ 10)
                    .attr('height', (s) => height - yScale(s.value) + 10)
                    .style("transform", "scale(1,0.979)"); 

                // Reference line for y values of rect    
                d3.select('svg').select('g')
                .append('line')
                .attr('id','ref-line')
                .attr('x1', 0)
                .attr('y1', yScale(s.value))
                .attr('x2', width)
                .attr('y2', yScale(s.value))
                .attr('transform','translate('+margin+',0)')
                .attr("stroke-width", 1)
                .attr("stroke", "red");

                // Frequency for the given range on the right side
                d3.select('svg').select('g')
                .append('text')
                .attr('id','ref-text')
                .attr('x', width + margin + 5)
                .attr('y', yScale(s.value))
                .style('fill','white')
                .text(s.value);
              
            })
            .on('mouseleave', function (actual, i) {

                // Restore the bar width and height on the mouseleave event
                d3.select(this)
                .attr("opacity", 0.8)
                .transition()
                .duration(200)
                .attr('x', (s) => enhanced_xscale(s.key) + margin)
                .attr('y', (s) => yScale(s.value))
                .attr('width',rectWidth)
                .attr('height', (s) => height - yScale(s.value))
                .style("transform", "scale(1,1)");

                // Remove ref line and text
                d3.select('#ref-line').remove();
                d3.select('#ref-text').remove();

            })

    const t = d3.transition()
      .duration(750);

    chart.selectAll('rect')
    .transition(t)
    .attr('height', (s) => height - yScale(s.value))
    .attr('y', (s) => yScale(s.value));
           
    // x-axis label
    svg.append('text')
    .attr('y', (height) + 2*margin - 20)
    .attr('x', width / 2 + margin)
    .attr('text-anchor', 'middle')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size', label_size)
    .style('letter-spacing',letter_spacing)
    .text(labels[feature_name]['x']);

     // title
    svg.append('text')
    .attr('x', width / 2 + margin)
    .attr('y', 40)
    .attr('text-anchor', 'middle')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',label_size)
    .style('letter-spacing',letter_spacing)
    .text(labels[feature_name]['title'])

    // y-axis label
    svg.append('text')
    .attr('x', - height/1.5)
    .attr('y', margin / 2.4 + 50)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',label_size)
    .style('letter-spacing',letter_spacing)
    .text(labels[feature_name]['y']);

     // y axis marker
    svg.append('text')
    .attr('y', 70 )
    .attr('x', -80) 
    .attr('transform', 'rotate(-90)')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',"10px")
    .text('y -->')

    // x axis marker
     svg.append('text')
    .attr('x', width + 90)
    .attr('y', height + 100) 
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',"10px")
    .text('x -->')

 }


 function axis_scaler(nBin, range, domain){
    var phy_unit = Math.ceil(range[1] / domain[1]);
    bandwidth = Math.ceil(domain[1] / nBin);

    var xMap = {};
    for(i=domain[0];i<domain[1];i+=bandwidth){
        xMap[parseInt(i)] = phy_unit*parseInt(i);
    }   
  
    return xMap;
 }


// Render the bar graph
 function renderBarGraph(feature_name, data){

    // Clear existing graph
    d3.selectAll("svg").remove();

    const width = 800;
    const height = 600 - 2 * margin;

    // Getting counts for unique values of features
    var sample = d3.nest()
                .key(function(d) { return d[feature_name]; })
                .sortKeys(d3.ascending)
                .rollup(function(leaves) { return leaves.length; })
                .entries(data)

    var max_val_x= d3.max(sample, function(d) { return d.value;} );

    // Append svg in graph area
    var which = 0;
    var svg = d3.select("#graph_area")
    .append("svg")
    .attr("width", "73.75em")
    .style("height", "42em")  
    .style("border", "1px solid")
    .append("g")
    .attr("transform", "translate(" + margin + "," + margin + ")"); 

    // Mouse drag prevent in bar graph
    d3.select("svg")
    .on("mousedown", function() {
        which = 1;
    })
    .on("mouseup", function() {
        document.body.style.cursor = "initial";
        which = 0;
    })
    .on("mouseout", function() {
        document.body.style.cursor = "initial";
        which = 0;
    })
    .on("mousemove", function(){
        // Left click
        if(which == 1)
            document.body.style.cursor = "not-allowed";
        
    });
    
    // Color schema for the bars
    var colorSchema = d3.scaleOrdinal()
       .domain(sample.map((s) => s.key))
       .range(d3.schemeSet3);

    // Tip added on bar hover
    var tip = d3.tip()
      .attr('class', 'd3-tip')
      .offset([-10, 0])
      .html(function(d) {      
          return "<span style='text-align:center;color:"+colorSchema(d.key)+"'> Value<sub>x</sub> - " + d.key + "<br> Frequency - " + d.value + "</span>";
      })

    svg.call(tip);

    // Linear y scale 
     yScale = d3.scaleLinear()
    .range([height, 0])
    .domain([0, max_val_x+ 50]);

    // Y scale append
    svg.append('g')
    .attr("transform", "translate(" + margin + ", 0)")
    .transition()
    .duration(axis_transition_time)
    .call(d3.axisLeft(yScale));

    // Scale banding the x scale for categorical data
     xScale = d3.scaleBand()
    .range([0, width])
    .domain(sample.map((s) => s.key))
    .padding(0.2)

    // Append the x axis
    svg.append('g')
    .attr('transform', `translate(`+margin+`, ${height})`)
    .transition().duration(axis_transition_time)
    .call(d3.axisBottom(xScale))
    .selectAll('text')
    .style("text-anchor", "end")
    .attr("dx", "-1em")
    .attr("dy", "-.3em")
    .attr("transform", "rotate(-65)");


    // Append bars on svg
    var bars = svg.selectAll()
            .data(sample)
            .enter()
            .append('rect')
            .attr('x', (s) => xScale(s.key)+margin)
            .attr('y', height)
            .attr('height', 0)
            .attr('width', xScale.bandwidth())
            .attr("fill", (s) => colorSchema(s.key))
            .on('mouseover', tip.show)
            .on('mouseout', tip.hide)
            .on('mouseenter', function (s, i) {

                d3.select(this)
                .transition()
                .duration(200)
                .attr('opacity', 0.6)
                .attr('x', (s) => xScale(s.key) +margin -5)
                .attr('y', (s) => yScale(s.value))
                .attr('width', xScale.bandwidth() + 10)
                .attr('height', (s) => height - yScale(s.value) + 10)
                .style("transform", "scale(1,0.979)"); 

                // Reference line for y values of rect    
                d3.select('svg').select('g')
                .append('line')
                .attr('id','ref-line')
                .attr('x1', 0)
                .attr('y1', yScale(s.value))
                .attr('x2', width)
                .attr('y2', yScale(s.value))
                .attr('transform','translate('+margin+',0)')
                .attr("stroke-width", 1)
                .attr("stroke", "red");

                d3.select('svg').select('g')
                .append('text')
                .attr('id','ref-text')
                .attr('x', width + margin + 5)
                .attr('y', yScale(s.value))
                .style('fill','white')
                .text(s.value);

            })
            .on('mouseleave', function (actual, i) {
                d3.select(this)
                .attr("opacity", 1)
                .transition()
                .duration(200)
                .attr('x', (s) => xScale(s.key) +margin)
                .attr('y', (s) => yScale(s.value))
                .attr('opacity', 1)
                .attr('width', xScale.bandwidth())
                .attr('height', (s) => height - yScale(s.value))
                .style("transform", "scale(1,1)")

                // Remove ref line
                d3.select('#ref-line').remove();
                d3.select('#ref-text').remove();
            })

    const t = d3.transition()
      .duration(axis_transition_time);

    svg.selectAll('rect')
    .transition(t)
    .attr('height', (s) => height - yScale(s.value))
    .attr('y', (s) => yScale(s.value));

    // Labels on axis

   
    // x-axis label
    svg.append('text')
    .attr('y', (height) + margin + 20)
    .attr('x', width / 2 + margin)
    .attr('text-anchor', 'middle')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',label_size)
    .style('letter-spacing',letter_spacing)
    .text(labels[feature_name]['x']);

    // title
    svg.append('text')
    .attr('x', width / 2 + margin)
    .attr('y', -20)
    .attr('text-anchor', 'middle')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',label_size)
    .style('letter-spacing',letter_spacing)
    .text(labels[feature_name]['title'])

    // y-axis label
    svg.append('text')
    .attr('x', - height/2)
    .attr('y', margin / 2.5 )
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',label_size)
    .style('letter-spacing','3')
    .text(labels[feature_name]['y']);


    // y axis marker
    svg.append('text')
    .attr('y', 20 )
    .attr('x', -20) 
    .attr('transform', 'rotate(-90)')
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',"10px")
    .text('y -->')

    // x axis marker
     svg.append('text')
    .attr('x', width + 40)
    .attr('y', height + 50) 
    .attr('fill','white')
    .style('stroke','white')
    .style('font-size',"10px")
    .text('x -->')
    
}