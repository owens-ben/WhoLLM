import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Roomer",
                systemImage: "house",
                description: Text("Know who's in which room.")
            )
            .navigationTitle("Roomer")
        }
    }
}

#Preview {
    ContentView()
}
