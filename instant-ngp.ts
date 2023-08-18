import * as fs from "fs";
import {Value} from "./value.js";
import {PNG} from "pngjs";
import {clamp, dim, PNGImage, render, saveImage, ThePNGImage} from "./render.js";
import {NgpMLP} from "./hash-mlp.js";
// neural graphics primitives

// what we need:
// 1. PNG writer (grayscale at first)
// 2. a picture to train on
// 3. a neural network
// 4. spatial hash encoder
// 5. a way to render the network
// 6. a way to train the network

function loopC() {
  let input = PNG.sync.read(fs.readFileSync('cameraman.png'));
  let image = new PNGImage(256,256,256,'black') as ThePNGImage;
  let grays = [];
  for(let i=0;i<256;i++){grays.push(image.createColor(`rgb(${i},${i},${i})`));}

  let nn = new NgpMLP(2,[64,128,256,512,512],[8,16,32,64,128],[15,10,1]);
  console.log({dim, params: nn.parameters().length});

  let chosenFn = (x, y) => {
    let x2 = x * input.width | 0;
    let y2 = y * input.height | 0;
    let idx = (input.width * y2 + x2) << 2;
    let red = input.data[idx];
    return red / 255; // it's grayscale
  };
  // reference
  let sz = dim;
  for(let y=0;y<sz;y++){
    for(let x=0;x<sz;x++){
      let expected = chosenFn(x / sz, y / sz); // normalized
      image.setPixel(x, y + dim, grays[clamp(Math.floor(expected * 256), 0, 255) | 0])
    }
  }
  let mul=1;
  let t = Date.now();
  let idt=0;
  let prevAvg=1;
  // train
  for(let i=0;i<10000*mul;i++){
    let maxLoss = 0;let avgLoss = 0; let avgLossN = 0;
    for(let y=0;y<sz;y++){
      for(let x=0;x<sz;x++){
        let input = [x/sz,y/sz]; // normalized
        let expected = chosenFn(input[0],input[1]);
        let output = nn.forward(input);
        let loss = (output as Value).sub(expected).abs().pow(2);
        maxLoss = Math.max(maxLoss,loss.data);
        avgLoss += loss.data; avgLossN++;
        loss.backward();
        for(let p of nn.parameters()){
          p.data -= 0.005*p.grad; p.grad = 0;
        }
      }
      // y+=(Math.random()*15)|0; // skip some lines
    }
    let dt = Date.now()-t;
    if(dt>5000){ // every 5 seconds
      idt+=1;
      render(nn, grays, image);
      let name = `loopC-${(''+idt).padStart(6,'0')}.png`;
      saveImage(image,`out/all/${name}`);
      fs.copyFileSync(`out/all/${name}`,`out/loopC.png`);
      if(idt%10==1){fs.copyFileSync(`out/all/${name}`,`out/10/loopC-${(''+(idt/10|0)).padStart(6,'0')}.png`);}
      if(idt%100==1){fs.copyFileSync(`out/all/${name}`,`out/100/loopC-${(''+(idt/100|0)).padStart(6,'0')}.png`);}

      avgLoss/=avgLossN;let improv = prevAvg/avgLoss;prevAvg=avgLoss;
      console.log(JSON.stringify({i,idt,
        max:maxLoss.toFixed(3),
        avg:avgLoss.toFixed(6),
        improv:improv.toFixed(3),
        dt:(dt*0.001).toFixed(1)+'s',
      }).replaceAll('"','').replaceAll(",",", "));
      t = Date.now();
    }
  }
}

loopC();