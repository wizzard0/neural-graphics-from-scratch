import {expect, test} from "bun:test";
import {Param, Value} from "./value.js";
import {MLP} from "./mlp.js";

test("3-4-4-1 MLP", () => {
  let {loss,pred} = mlpScenario();
//  console.log({pp:preds[0][0]}, preds[0].data)//.map(x=>x.toString()));
  expect(pred[0]).toBeCloseTo(1.0, 1)
  expect(pred[1]).toBeCloseTo(-1.0, 1)
  expect(pred[2]).toBeCloseTo(-1.0, 1)
  expect(pred[3]).toBeCloseTo(1.0, 1)
});

function mlpScenario(): {loss:number,pred:number[]} {
  let log = console.log
  let n = new MLP(3, [4, 4, 1])

  let inputSets = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]

  let outputYs = [1.0, -1.0, -1.0, 1.0]
  let last_pred: Value[] = []

  function train(round) {

    let ypred: Value[] = []
    inputSets.forEach(xInput => {
      ypred.push(n.forward(xInput) as Value)
    })

    let loss_total = new Value(0.0, '', 'loss_total');
    //console.log(ys.length, ypred.length);
    for (let i = 0; i < outputYs.length; i++) {
      let loss = new Value(outputYs[i], '', 'loss');
      loss = loss.sub(ypred[i]).pow(2)
      loss_total = loss_total.add(loss)
    }

    loss_total.backward() // does zero grad inside
    let rate = 0.01// 3/(round*round+10), also 0.05 is too much for relu

    for (let p of n.parameters()) {
      p.data += -rate * p.grad
    }
//    console.log(round, ypred.map(x=>x.toString()));
    if (round % 100 == 99 || round < 10) {
    //  log(round, rate.toFixed(6), "loss=" + loss_total.data.toFixed(4))
    //  log(ypred.map(x => x.data.toFixed(4)))
    }
    last_pred = ypred
    return loss_total
  } // train

  let loss:Value;
  for (let i = 0; i < 150; i++) {
    loss=train(i)
  }

  // log("Predictions after training", n.parameters().length)
  // for (let p of last_pred) {
  //   log(p.data)
  // }
  return {loss:loss.data, pred:last_pred.map(x=>x.data)};
}