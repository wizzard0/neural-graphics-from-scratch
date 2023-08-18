export function ensureValue(other_: number | Value) {
  // slow. all of this is slow.
  if(typeof other_ === 'number'){
    return new Value(other_, 'input', ''+other_);
  }else if (typeof other_?.grad === 'number') {
    return other_;
  }else if(Array.isArray(other_) && other_.length === 1 && typeof other_[0]?.grad === 'number'){
    return other_[0];
  } else {
    console.log({other_}); throw new Error('not a value');
  }
}

// differentiable value, modeled after micrograd
export class Value {
  data: number;
  a: Value;
  b: Value;
  op: string;
  grad: number;
  name:string;
  constructor(data:number, op:string, name:string, a?:Value, b?:Value) {
    if(isNaN(data)){
      console.log({data});
      throw new Error('data is NaN');
    }
    this.data = data;
    this.op = op;
    this.a=a;
    this.b=b;
    this.grad = 0;
    this.name = name;
  }

  add(other_: number|Value, name?: string): Value {
    return new SumValue(this, ensureValue(other_), name);
  }

  mul(other_: number|Value, name?: string): Value {
    return new MulValue(this, ensureValue(other_),name);
  }

  div(other_: number|Value, name?: string): Value {
    return new DivValue(this, ensureValue(other_),name);
  }

  pow(other_: number|Value, name?: string): Value {
    if(typeof other_ === 'number'){
      return new PowConstValue(this, other_,name);
    }else{
      return new PowValue(this, other_,name);
    }
  }

  relu(leak=0,name?:string): Value {
    return new ReLU(this,leak, name);
  }


  _backward() {
  }

  backward(){
    let visited:Set<Value> = new Set();
    let queue:Value[] = [this];
    let result:Value[] = [];

    while(queue.length){
      let node = queue.shift();
      if(visited.has(node)){continue;}
      visited.add(node);
      result.push(node);
      node.a&&queue.push(node.a);
      node.b&&queue.push(node.b);
    }
    this.grad = 1;
    for(let node of result){
      node._backward();
    }
  }

  toString() {
    return `${this.name||""}=${this.op}(${this.data.toFixed(4)}, ${this.grad.toFixed(4)})`;
  }

  exp(name?:string): Value {
    return new Exp(this,name);
  }

  abs(name?:string): Value {
    return new Abs(this,name);
  }

  sub(other_: number|Value, name?: string) {
    return this.add(ensureValue(other_).mul(-1),name);
  }

  tanh(name?:string): Value {
    let e2x = this.mul(2).exp();
    return (e2x.sub(1)).div(e2x.add(1),name);
  }
}

export class Param extends Value {
  constructor(data:number, name:string) {
    super(data, 'param', name);
  }

  _backward() {
  }
}

class SumValue extends Value {
  constructor(a: Value, b: Value, name: string) {
    super(a.data+b.data, '+', name, a,b);
  }

  _backward() {
    this.a.grad += this.grad;
    this.b.grad += this.grad;
  }
}

class MulValue extends Value {
  constructor(a: Value, b: Value, name: string) {
    super(a.data*b.data, '*', name,a,b);
  }

  _backward() {
    this.a.grad += this.b.data*this.grad;
    this.b.grad += this.a.data*this.grad;
  }
}

class DivValue extends Value {
  constructor(a: Value, b: Value, name: string) {
    super(a.data/b.data, '/',  name,a,b);
  }

  _backward() {
    this.a.grad += this.grad / this.b.data;
    this.b.grad += -this.grad * this.a.data / (this.b.data * this.b.data);
  }
}

class PowValue extends Value {
  constructor(a: Value, b: Value, name: string) {
    super(Math.pow(a.data,b.data), '^',  name,a,b);
  }

  _backward() {
    this.a.grad += this.b.data * Math.pow(this.a.data, this.b.data - 1) * this.grad;
    this.b.grad += Math.pow(this.a.data, this.b.data) * Math.log(this.a.data) * this.grad;
  }
}

class PowConstValue extends Value {
  _b: number;
  constructor(a: Value, b: number, name: string){
    super(Math.pow(a.data,b), '^', name,a);
    this._b = b;
  }

  _backward() {
    this.a.grad += (this._b*Math.pow(this.a.data,this._b-1))*this.grad;
  }
}

class ReLU extends Value {
  leak: number;
  constructor(a: Value,leak:number, name: string) {
    super(a.data>0?a.data:a.data*leak, 'ReLU',name, a);
    this.leak = leak;
  }

  _backward() {
    this.a.grad += this.a.data > 0 ? this.grad : this.grad * this.leak;
  }
}

class Abs extends Value {
  constructor(a: Value,name: string) {
    super(Math.abs(a.data), 'Abs',name, a);
  }

  _backward() {
    this.a.grad += this.a.data >= 0 ? this.grad : -this.grad;
  }
}

class Exp extends Value {
  constructor(a: Value, name: string) {
    super(Math.exp(a.data), 'exp',name, a);
  }

  _backward() {
    this.a.grad += Math.exp(this.a.data)*this.grad;
  }
}