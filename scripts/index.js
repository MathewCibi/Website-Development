
const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

window.addEventListener('load', function () {
  const enhance = id => {
    console.log(document.getElementById("channel_link"))
    const element = document.getElementById(id),
          text = element.innerText.split("");
          console.log(text)
document.addEventListener('DOMContentLoaded', function() {
  const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
  const enhance = id => {
    const element = document.getElementById(id),
          text = element.innerText.split("");
    
    element.innerText = "";
    
    text.forEach((value, index) => {
      const outer = document.createElement("span");
      
      outer.className = "outer";
      
      const inner = document.createElement("span");
      
      inner.className = "inner";
      
      inner.style.animationDelay = `${rand(-5000, 0)}ms`;
      
      const letter = document.createElement("span");
      
      letter.className = "letter";
      
      letter.innerText = value;
      
      letter.style.animationDelay = `${index * 1000 }ms`;
      
      inner.appendChild(letter);    
      
      outer.appendChild(inner);    
      
      element.appendChild(outer);
    });
    
  }

  enhance("channel_link");
})


  }
  
  enhance("channel_link");
}, false);


