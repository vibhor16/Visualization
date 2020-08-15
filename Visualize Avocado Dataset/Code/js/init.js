var columnNames = [];
var categorical_features = ['Type','Year','City','Rating','Major Size','Population'];
var numerical_features = ['Price','Total Volume','Total Bags','PLU4046','PLU4225','PLU4770','Small Bags','Large Bags','XLarge Bags'];

var slider = document.getElementById("theSlider");
var select_list;
var nBin;
var DATA;
var FEATURE_NAME;
var defaultSliderVal = 10;
var maxSlider = 20;
var minSlider = 1;
var delayInMilliseconds = 1000;
var effect_timeout = 500;

$(document).ready(function(){

  // Hide feature menu in the beginning
  $("#feature_menu").hide();
  $("#home_text").hide();
  
  populateFeatureMenu();
  // Show feature menu
  $("#vibrating-title").click(function(){
    $(this).attr('id','');  // Stop vibrating
    show_menu();
    $("#graph_area").hide(effect_timeout);
    $("#theSlider").hide(effect_timeout);
    $("#home_text").toggle();
  });
  
});

function populateFeatureMenu(){
  
  $("#feature_menu").html("");

  for(var i=0;i<categorical_features.length;i++){
    var content = '<div class="w3-quarter feature" style="cursor: pointer;width: 20%; ">'+
      '<div class="w3-container w3-padding-4 feature-cell" value="' + categorical_features[i] + '" onclick="display(this)">'+
        '<div class="w3-left"><h4>' + categorical_features[i] +'</h4></div>'+
        '<div class="w3-right">'+
          '<p><b style="color: orange;font-size: 9px;vertical-align: -webkit-baseline-middle;">CATEGORICAL</b></p>'+
        '</div>'+
        '<div class="w3-clear"></div>'+
      '</div>'
    '</div>';

    $("#feature_menu").append(content);
  }
  for(var i=0;i<numerical_features.length;i++){
    var content = '<div class="w3-quarter feature" style="cursor: pointer;width: 20%">'+
      '<div class="w3-container w3-padding-4 feature-cell" value="'+ numerical_features[i] +'" onclick="display(this)">'+
        '<div class="w3-left"><h4>' + numerical_features[i] + '</h4></div>'+
        '<div class="w3-right">'+
          '<p><b style="color: lawngreen;font-size: 9px;vertical-align: -webkit-baseline-middle;">NUMERICAL</b></p>'+
        '</div>'+
        '<div class="w3-clear"></div>'+
      '</div>'
    '</div>';
    $("#feature_menu").append(content);
    
  }
}


function show_menu(){
  // visible
  if($("#feature_menu").is(":visible")){

    if($(".highlight-selected").length > 0){
      var parent = $(".highlight-selected").parent();
      populateFeatureMenu();
      $($(parent).children()[0]).addClass("highlight-selected");
    } else {
      $("#feature_menu").hide(effect_timeout);
    }
    
  } else {
    //not visible
    $("#feature_menu").show(effect_timeout);
  }
  
}

function hide_other_menu_options(){
  $.each( $(".feature"), function(id, obj){
    if(!$(obj).children()[0].classList.contains("highlight-selected")){
      $(obj).hide(effect_timeout);
    }
  });
}

function display(el){
  
  $("#home_text").hide();
  $("#graph_area").show(effect_timeout);

  FEATURE_NAME = el.getAttribute("value");
  toggleHighlight(el);
  var home_text = document.getElementById("home_text");

  if(home_text !== null && home_text !== undefined)
    home_text.remove();

  d3.csv("dataset/avocado_dataset.csv").then(function(data) {
    DATA = data;
   
    if(categorical_features.includes(FEATURE_NAME)){
              renderGraph(FEATURE_NAME, DATA, "CATEGORICAL");
          } else  {
              resetSlider(defaultSliderVal);
              $("#theSlider").show(effect_timeout);
              renderGraph(FEATURE_NAME, DATA, "NUMERICAL",defaultSliderVal);
     }
  

  });
}

function toggleHighlight(el){
  var highlightedOptions = document.getElementsByClassName("highlight-selected");

  if(highlightedOptions.length > 0){
    highlightedOptions[0].classList.remove("highlight-selected");
  }
  el.classList.add("highlight-selected");
  hide_other_menu_options();


}
