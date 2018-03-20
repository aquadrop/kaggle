package leetcode;

import java.util.Stack;

public class LC232 {

}

class MyQueue {

    /** Initialize your data structure here. */
    Stack<Integer> stack;
    public MyQueue() {
        this.stack = new Stack<Integer>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        Stack<Integer> tmp = new Stack<Integer>();
        while (!this.stack.isEmpty()) {
        	tmp.push(this.stack.pop());
        }
        
        this.stack.push(x);
        
        while(!tmp.isEmpty()) {
        	this.stack.push(tmp.pop());
        }
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        return this.stack.pop();
    }
    
    /** Get the front element. */
    public int peek() {
        return this.stack.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return this.stack.isEmpty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */
