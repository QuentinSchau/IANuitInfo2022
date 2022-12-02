import React from "react";
import './App.css';
import {searchDefinitionWord} from './Algorithmes.js';

class App extends React.Component {

  // Constructor
  constructor(props) {
    super(props);

    this.state = {
      items: [],
      DataisLoaded: false
    };
  }

  // ComponentDidMount is used to
  // execute the code
  componentDidMount() {
    fetch(
        "https://api.orphacode.org/FR/ClinicalEntity", {
                headers: {
                    'apiKey': 'grinch'
                }
        })
        .then((res) => res.json())
        .then((json) => {
          this.setState({
            items: json,
            DataisLoaded: true
          });
        })
  }
  render() {
    const { DataisLoaded, items } = this.state;
    if (!DataisLoaded) return <div>
      <h1> Pleases wait some time.... </h1> </div> ;

    return (
        <div className = "App">
          <h1> SFEIR EST - Grinch </h1>
            <div className="topnav">
            <input type="text" id="myInput" onKeyUp={searchDefinitionWord} placeholder="Search for term.." />
            <a className="active" href="App.js">Home</a>
            </div>
            <table id="myTable">
                <thead>
                <tr>
                    <th className="th-sm">ORPHAcode</th>
                    <th className="th-sm">Term</th>
                    <th className="th-sm">Definition</th>
                </tr>
                </thead>
                <tbody>
                {
                    items.map((item) => (
                      <tr key = { item.ORPHAcode }>
                        <td>{ item["ORPHAcode"] }</td>
                        <td>{ item["Preferred term"] }</td>
                        <td>{ item["Definition"] }</td>
                      </tr>
                  ))
                }
                </tbody>
            </table>
        </div>
    );
  }
}

export default App;
